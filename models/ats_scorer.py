"""
models/ats_scorer.py
────────────────────────────────────────────────────────────────────────────────
ATS Score Prediction Engine
────────────────────────────────────────────────────────────────────────────────
Implements a multi-layer scoring pipeline:

  Layer 1 — TF-IDF Cosine Similarity  (fast baseline)
  Layer 2 — Sentence-BERT Embeddings  (semantic matching)
  Layer 3 — Structural / Heuristic    (formatting, completeness, action verbs)
  Layer 4 — Ensemble                  (weighted combination → 0-100 score)

Sub-scores returned:
  • keyword_score       : TF-IDF + BERT keyword overlap with JD
  • formatting_score    : Section completeness, bullet usage, length
  • completeness_score  : Presence of required sections
  • readability_score   : Word variety, sentence length, clarity
  • action_verb_score   : Presence and density of action verbs
────────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import os
import numpy as np
import joblib
from dataclasses import dataclass, asdict
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# ── Constants ────────────────────────────────────────────────────────────────
ACTION_VERBS = [
    "developed", "designed", "implemented", "built", "optimized", "led",
    "managed", "delivered", "achieved", "reduced", "improved", "increased",
    "automated", "architected", "launched", "created", "deployed", "analysed",
    "coordinated", "established", "executed", "generated", "integrated",
    "spearheaded", "streamlined", "collaborated", "engineered", "mentored",
]

REQUIRED_SECTIONS = ["skills", "experience", "education", "summary"]
PREFERRED_SECTIONS = ["projects", "certifications", "contact"]

FILLER_WORDS = [
    "responsible for", "worked on", "helped with", "assisted in",
    "involved in", "participated in", "was part of",
]

QUANTIFICATION_RE = re.compile(r"\d+\s*(%|percent|x|times|users|months|years|k\b)")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ats_scorer_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")


@dataclass
class ATSScore:
    overall_score: float
    keyword_score: float
    formatting_score: float
    completeness_score: float
    readability_score: float
    action_verb_score: float
    missing_keywords: list
    matched_keywords: list
    improvement_tips: list
    sub_score_weights: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def grade(self) -> str:
        if self.overall_score >= 80:
            return "Excellent"
        elif self.overall_score >= 65:
            return "Good"
        elif self.overall_score >= 50:
            return "Average"
        else:
            return "Needs Improvement"


class TFIDFScorer:
    """Fast TF-IDF based keyword scoring."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus: list):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def score(self, resume_text: str, jd_text: str) -> tuple[float, list, list]:
        """Returns (similarity_score 0-100, matched_keywords, missing_keywords)."""
        if not self._fitted:
            self.vectorizer.fit([resume_text, jd_text])

        vectors = self.vectorizer.transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Keyword overlap analysis
        jd_terms = set(self._extract_keywords(jd_text))
        resume_terms = set(self._extract_keywords(resume_text))

        matched = list(jd_terms & resume_terms)
        missing = list(jd_terms - resume_terms)

        return float(similarity * 100), matched, missing

    def _extract_keywords(self, text: str, top_n: int = 30) -> list:
        """Extract top TF-IDF keywords from text."""
        try:
            vec = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = vec.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except Exception:
            return text.lower().split()[:top_n]


class SemanticScorer:
    """
    Sentence-BERT semantic similarity scorer using all-MiniLM-L6-v2.

    Goes beyond simple keyword overlap — understands meaning:
      "built REST APIs" matches "backend web services" even with zero shared words.

    Three-layer semantic analysis:
      1. Full-document cosine similarity   (overall semantic alignment)
      2. Per-skill semantic matching        (which JD skills semantically exist in resume)
      3. Section-weighted scoring           (experience/skills sections weighted higher)

    Falls back gracefully to TF-IDF if sentence-transformers not installed.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, fast CPU inference, strong performance
    SEMANTIC_THRESHOLD = 0.45          # cosine similarity threshold for skill match

    def __init__(self):
        self.model = None
        self._cache = {}               # embedding cache to avoid recomputation
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.MODEL_NAME)
            print(f"  ✓ Sentence-BERT loaded ({self.MODEL_NAME}) — semantic scoring active")
        except ImportError:
            print("  ⚠  sentence-transformers not installed.")
            print("     Run: pip install sentence-transformers")
            print("     Falling back to TF-IDF similarity.")
            self.model = None
        except Exception as e:
            print(f"  ⚠  Sentence-BERT load failed: {e}. Using TF-IDF fallback.")
            self.model = None

    @property
    def is_bert_active(self) -> bool:
        return self.model is not None

    def _embed(self, texts: list) -> "np.ndarray":
        """Encode texts with caching."""
        results = []
        to_encode = []
        indices = []
        for i, t in enumerate(texts):
            key = hash(t[:500])
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                to_encode.append(t)
                indices.append(i)

        if to_encode:
            new_embeddings = self.model.encode(to_encode, show_progress_bar=False,
                                                batch_size=32, normalize_embeddings=True)
            for idx, emb in zip(indices, new_embeddings):
                self._cache[hash(texts[idx][:500])] = emb
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])

    def score(self, resume_text: str, jd_text: str) -> float:
        """
        Returns overall semantic similarity score 0-100.
        Uses BERT if available, falls back to TF-IDF.
        """
        if self.model is None:
            return self._tfidf_fallback(resume_text, jd_text)
        return self._bert_score(resume_text, jd_text)

    def _bert_score(self, resume_text: str, jd_text: str) -> float:
        """
        Three-layer BERT scoring:
          40% — full document cosine similarity
          40% — per-skill semantic match rate
          20% — key section similarity (experience + skills)
        """
        # Layer 1: Full document similarity
        r_trunc = resume_text[:3000]
        j_trunc = jd_text[:3000]
        doc_embs = self._embed([r_trunc, j_trunc])
        doc_sim = float(cosine_similarity([doc_embs[0]], [doc_embs[1]])[0][0])

        # Layer 2: Per-skill semantic matching
        jd_skills = self._extract_skill_phrases(jd_text)
        skill_match_rate = 0.0
        if jd_skills:
            resume_sentences = [s.strip() for s in resume_text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
            if resume_sentences:
                skill_embs = self._embed(jd_skills[:20])
                resume_embs = self._embed(resume_sentences[:40])
                sim_matrix = cosine_similarity(skill_embs, resume_embs)
                # For each JD skill, check if any resume sentence matches
                matched = sum(1 for i in range(len(jd_skills[:20]))
                              if sim_matrix[i].max() >= self.SEMANTIC_THRESHOLD)
                skill_match_rate = matched / len(jd_skills[:20])

        # Layer 3: Section-weighted similarity (experience vs full doc)
        resume_exp = self._extract_section(resume_text, ["experience", "work"])
        jd_req = self._extract_section(jd_text, ["required", "responsibilities", "skills"])
        section_sim = 0.0
        if resume_exp and jd_req:
            sec_embs = self._embed([resume_exp[:1500], jd_req[:1500]])
            section_sim = float(cosine_similarity([sec_embs[0]], [sec_embs[1]])[0][0])

        # Weighted ensemble
        final = (doc_sim * 0.40) + (skill_match_rate * 0.40) + (section_sim * 0.20)
        return float(np.clip(final * 100, 0, 100))

    def get_semantic_keyword_matches(self, resume_text: str, keywords: list) -> dict:
        """
        For each keyword, find the most semantically similar phrase in the resume.
        Returns dict: {keyword: {"matched": bool, "best_match": str, "score": float}}
        Useful for near-miss detection beyond simple string overlap.
        """
        if self.model is None or not keywords:
            return {}

        resume_phrases = [s.strip() for s in resume_text.replace('\n', '. ').split('.')
                          if len(s.strip()) > 8][:50]
        if not resume_phrases:
            return {}

        kw_embs = self._embed(keywords[:30])
        phrase_embs = self._embed(resume_phrases)
        sim_matrix = cosine_similarity(kw_embs, phrase_embs)

        results = {}
        for i, kw in enumerate(keywords[:30]):
            best_idx = sim_matrix[i].argmax()
            best_score = float(sim_matrix[i][best_idx])
            results[kw] = {
                "matched": best_score >= self.SEMANTIC_THRESHOLD,
                "best_match": resume_phrases[best_idx][:80],
                "score": round(best_score, 3)
            }
        return results

    def _extract_skill_phrases(self, text: str) -> list:
        """Extract likely skill/requirement phrases from JD."""
        lines = [l.strip() for l in text.replace(',', '\n').split('\n') if l.strip()]
        skills = []
        for line in lines:
            # Keep short phrases (likely skill names)
            if 1 <= len(line.split()) <= 6 and len(line) > 2:
                skills.append(line)
        return skills[:30]

    def _extract_section(self, text: str, keywords: list) -> str:
        """Extract a section from text by keyword headers."""
        lines = text.split('\n')
        section_lines = []
        in_section = False
        for line in lines:
            ll = line.lower().strip()
            if any(kw in ll for kw in keywords) and len(ll) < 40:
                in_section = True
                continue
            if in_section:
                if ll and len(ll) < 30 and ll.isupper():
                    break  # hit next section header
                section_lines.append(line)
                if len(section_lines) > 20:
                    break
        return ' '.join(section_lines)

    def _tfidf_fallback(self, resume_text: str, jd_text: str) -> float:
        """TF-IDF cosine similarity fallback when BERT unavailable."""
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=3000)
        try:
            vectors = vec.fit_transform([resume_text, jd_text])
            return float(np.clip(
                cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 0, 100
            ))
        except Exception:
            return 50.0


class HeuristicScorer:
    """Rule-based structural and formatting scorer."""

    def score_formatting(self, resume_text: str) -> float:
        score = 50.0

        # Length check (ideal: 300-800 words)
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:
            score += 20
        elif 200 <= word_count < 300 or 800 < word_count <= 1200:
            score += 10
        else:
            score -= 10

        # Bullet points
        bullet_count = resume_text.count("•") + resume_text.count("-") + resume_text.count("*")
        if bullet_count >= 5:
            score += 15
        elif bullet_count >= 2:
            score += 8

        # No filler words
        text_lower = resume_text.lower()
        filler_count = sum(text_lower.count(f) for f in FILLER_WORDS)
        score -= filler_count * 5

        # Quantification
        quantified = QUANTIFICATION_RE.findall(resume_text)
        score += min(len(quantified) * 5, 15)

        return float(np.clip(score, 0, 100))

    def score_completeness(self, resume_text: str) -> float:
        score = 0.0
        text_lower = resume_text.lower()

        # Required sections (15 pts each)
        required_found = 0
        for section in REQUIRED_SECTIONS:
            keywords = {
                "skills": ["skill", "technical", "competenc"],
                "experience": ["experience", "work", "employment"],
                "education": ["education", "degree", "university", "college"],
                "summary": ["summary", "objective", "profile", "about"],
            }[section]
            if any(kw in text_lower for kw in keywords):
                score += 15
                required_found += 1

        # Preferred sections (5 pts each)
        for section in PREFERRED_SECTIONS:
            keywords = {
                "projects": ["project"],
                "certifications": ["certif", "credential"],
                "contact": ["@", "phone", "email", "linkedin"],
            }[section]
            if any(kw in text_lower for kw in keywords):
                score += 5

        return float(np.clip(score, 0, 100))

    def score_readability(self, resume_text: str) -> float:
        score = 60.0
        sentences = [s.strip() for s in re.split(r"[.!?]", resume_text) if s.strip()]

        if not sentences:
            return score

        # Average sentence length (ideal: 10-20 words)
        avg_len = np.mean([len(s.split()) for s in sentences])
        if 10 <= avg_len <= 20:
            score += 20
        elif 7 <= avg_len < 10 or 20 < avg_len <= 30:
            score += 10
        else:
            score -= 10

        # Vocabulary richness
        words = resume_text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 20

        return float(np.clip(score, 0, 100))

    def score_action_verbs(self, resume_text: str) -> float:
        text_lower = resume_text.lower()
        found = [v for v in ACTION_VERBS if v in text_lower]
        score = min(len(found) * 10, 70) + 30
        return float(np.clip(score, 0, 100))


class ATSScorer:
    """
    Main ATS scoring engine combining TF-IDF, semantic, and heuristic scores.
    Supports training a meta-learner on labeled data.
    """

    WEIGHTS = {
        "keyword_score": 0.35,
        "formatting_score": 0.15,
        "completeness_score": 0.20,
        "readability_score": 0.10,
        "action_verb_score": 0.10,
        "semantic_score": 0.10,
    }

    def __init__(self):
        self.tfidf_scorer = TFIDFScorer()
        self.semantic_scorer = SemanticScorer()
        self.heuristic_scorer = HeuristicScorer()
        self.meta_model = None
        self._tfidf_fitted = False

    def fit_tfidf(self, corpus: list):
        """Fit TF-IDF vectorizer on a corpus of resume + JD texts."""
        self.tfidf_scorer.fit(corpus)
        self._tfidf_fitted = True

    def score(
        self,
        resume_text: str,
        jd_text: Optional[str] = None,
    ) -> ATSScore:
        """
        Score a resume against an optional job description.
        Returns ATSScore with all sub-scores and improvement tips.
        """
        jd = jd_text or ""

        # ── Sub-scores ────────────────────────────────────────────────────────
        if jd:
            keyword_score, matched_kw, missing_kw = self.tfidf_scorer.score(
                resume_text, jd
            )
            semantic_score = self.semantic_scorer.score(resume_text, jd)
        else:
            # General quality scoring without JD
            keyword_score = self._general_keyword_density(resume_text)
            matched_kw = []
            missing_kw = []
            semantic_score = keyword_score

        formatting_score = self.heuristic_scorer.score_formatting(resume_text)
        completeness_score = self.heuristic_scorer.score_completeness(resume_text)
        readability_score = self.heuristic_scorer.score_readability(resume_text)
        action_verb_score = self.heuristic_scorer.score_action_verbs(resume_text)

        # ── Ensemble ──────────────────────────────────────────────────────────
        if self.meta_model is not None:
            features = np.array([[
                keyword_score, formatting_score, completeness_score,
                readability_score, action_verb_score
            ]])
            overall = float(np.clip(self.meta_model.predict(features)[0], 0, 100))
        else:
            overall = (
                keyword_score * self.WEIGHTS["keyword_score"]
                + formatting_score * self.WEIGHTS["formatting_score"]
                + completeness_score * self.WEIGHTS["completeness_score"]
                + readability_score * self.WEIGHTS["readability_score"]
                + action_verb_score * self.WEIGHTS["action_verb_score"]
                + semantic_score * self.WEIGHTS["semantic_score"]
            )
            overall = float(np.clip(overall, 0, 100))

        # ── Improvement tips ──────────────────────────────────────────────────
        tips = self._generate_tips(
            resume_text, keyword_score, formatting_score,
            completeness_score, readability_score, action_verb_score,
            missing_kw[:10]
        )

        return ATSScore(
            overall_score=round(overall, 1),
            keyword_score=round(keyword_score, 1),
            formatting_score=round(formatting_score, 1),
            completeness_score=round(completeness_score, 1),
            readability_score=round(readability_score, 1),
            action_verb_score=round(action_verb_score, 1),
            missing_keywords=missing_kw[:15],
            matched_keywords=matched_kw[:15],
            improvement_tips=tips,
            sub_score_weights=self.WEIGHTS,
        )

    def train_meta_model(self, df):
        """
        Train a GBM meta-learner on labeled resume data.
        df must have columns: keyword_score, formatting_score, completeness_score,
                              readability_score, action_verb_score, ats_score
        """
        feature_cols = [
            "keyword_score", "formatting_score", "completeness_score",
            "readability_score", "action_verb_score"
        ]
        X = df[feature_cols].values
        y = df["ats_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Try multiple models, pick best
        models = {
            "GBM": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
            "RF": RandomForestRegressor(n_estimators=200, random_state=42),
            "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        }

        best_score = -np.inf
        best_model = None
        best_name = ""

        print("\n  Training meta-learner models:")
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            print(f"    {name}: R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
                best_name = name

        best_model.fit(X_train, y_train)
        train_r2 = best_model.score(X_train, y_train)
        test_r2 = best_model.score(X_test, y_test)

        print(f"\n  Best model: {best_name}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test  R²: {test_r2:.4f}")

        self.meta_model = best_model
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(self.tfidf_scorer.vectorizer, VECTORIZER_PATH)
        print(f"  ✓ Model saved to {MODEL_PATH}")

        return {"best_model": best_name, "train_r2": train_r2, "test_r2": test_r2}

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.meta_model = joblib.load(MODEL_PATH)
            print(f"  ✓ Meta-learner loaded from {MODEL_PATH}")
        if os.path.exists(VECTORIZER_PATH):
            self.tfidf_scorer.vectorizer = joblib.load(VECTORIZER_PATH)
            self.tfidf_scorer._fitted = True
            print(f"  ✓ TF-IDF vectorizer loaded from {VECTORIZER_PATH}")

    def _general_keyword_density(self, text: str) -> float:
        from utils.resume_parser import KNOWN_SKILLS
        text_lower = text.lower()
        found = sum(1 for s in KNOWN_SKILLS if s in text_lower)
        return float(np.clip((found / len(KNOWN_SKILLS)) * 200, 0, 100))

    def _generate_tips(
        self, resume_text, keyword_score, formatting_score,
        completeness_score, readability_score, action_verb_score, missing_kw
    ) -> list:
        tips = []
        text_lower = resume_text.lower()
        word_count = len(resume_text.split())

        # Keyword tips
        if keyword_score < 50 and missing_kw:
            tips.append({
                "category": "Missing Keywords",
                "issue": f"Your resume is missing {len(missing_kw)} important keywords.",
                "recommendation": f"Add these keywords naturally: {', '.join(missing_kw[:5])}",
                "expected_improvement": "+15-20 points",
                "priority": "High"
            })

        # Formatting tips
        if word_count < 250:
            tips.append({
                "category": "Formatting",
                "issue": f"Resume is too short ({word_count} words). ATS prefers 300-800 words.",
                "recommendation": "Expand experience descriptions and add a projects section.",
                "expected_improvement": "+10-15 points",
                "priority": "High"
            })
        elif word_count > 1000:
            tips.append({
                "category": "Formatting",
                "issue": f"Resume is too long ({word_count} words). Keep it concise.",
                "recommendation": "Trim to 1-2 pages (300-800 words) by removing irrelevant details.",
                "expected_improvement": "+5-10 points",
                "priority": "Medium"
            })

        # Completeness tips
        if "project" not in text_lower:
            tips.append({
                "category": "Section Gaps",
                "issue": "No Projects section found.",
                "recommendation": "Add 2-3 relevant projects with tech stack and outcomes.",
                "expected_improvement": "+8-12 points",
                "priority": "High"
            })
        if "certif" not in text_lower:
            tips.append({
                "category": "Section Gaps",
                "issue": "No Certifications section found.",
                "recommendation": "Add relevant certifications (AWS, Google, HackerRank, etc.).",
                "expected_improvement": "+5-8 points",
                "priority": "Medium"
            })
        if "summary" not in text_lower and "objective" not in text_lower:
            tips.append({
                "category": "Section Gaps",
                "issue": "No Summary/Objective section found.",
                "recommendation": "Add a 2-3 line professional summary at the top.",
                "expected_improvement": "+5-8 points",
                "priority": "Medium"
            })

        # Action verbs
        if action_verb_score < 50:
            tips.append({
                "category": "Action Verbs",
                "issue": "Weak or missing action verbs in experience descriptions.",
                "recommendation": "Start each bullet with strong verbs: Developed, Optimized, Led, Achieved, Delivered.",
                "expected_improvement": "+8-10 points",
                "priority": "Medium"
            })

        # Filler words
        filler_found = [f for f in FILLER_WORDS if f in text_lower]
        if filler_found:
            tips.append({
                "category": "Readability",
                "issue": f"Filler phrases detected: {', '.join(filler_found[:3])}",
                "recommendation": "Replace passive phrases with specific achievements and metrics.",
                "expected_improvement": "+5-8 points",
                "priority": "Medium"
            })

        # Quantification
        if not QUANTIFICATION_RE.search(resume_text):
            tips.append({
                "category": "Quantification",
                "issue": "No quantified achievements found.",
                "recommendation": "Add numbers to your impact: '...reduced load time by 40%', '...served 10k users'.",
                "expected_improvement": "+10-15 points",
                "priority": "High"
            })

        # LinkedIn
        if "linkedin" not in text_lower:
            tips.append({
                "category": "Formatting",
                "issue": "LinkedIn profile URL not found.",
                "recommendation": "Add your LinkedIn profile URL to the contact section.",
                "expected_improvement": "+3-5 points",
                "priority": "Low"
            })

        # Sort by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        tips.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return tips[:8]  # Return top 8


if __name__ == "__main__":
    scorer = ATSScorer()

    resume = """
    John Doe | john@email.com | linkedin.com/in/johndoe

    SUMMARY
    Data Scientist with 3 years of experience in Python, Machine Learning, and NLP.

    SKILLS
    Python, Machine Learning, TensorFlow, Scikit-learn, SQL, Pandas, AWS, Docker

    EXPERIENCE
    • Developed ML pipeline using Python and TensorFlow, reducing training time by 35%.
    • Implemented NLP model for text classification achieving 91% accuracy.
    • Deployed models on AWS SageMaker serving 5,000 daily requests.

    EDUCATION
    B.Tech, IIT Bombay, 2021

    CERTIFICATIONS
    AWS Certified Solutions Architect

    PROJECTS
    • Resume Screener: NLP-based ATS scoring using BERT and FAISS.
    """

    jd = """
    We are looking for a Data Scientist with:
    Required Skills: Python, Machine Learning, Deep Learning, SQL, TensorFlow, NLP,
    Pandas, Scikit-learn, Feature Engineering, Statistics
    Preferred: AWS, Docker, Spark, PyTorch
    Experience: 2+ years in Data Science
    """

    print("Scoring resume against job description...")
    result = scorer.score(resume, jd)
    print(f"\nOverall ATS Score: {result.overall_score}/100 ({result.grade()})")
    print(f"  Keyword Score:      {result.keyword_score}")
    print(f"  Formatting Score:   {result.formatting_score}")
    print(f"  Completeness Score: {result.completeness_score}")
    print(f"  Readability Score:  {result.readability_score}")
    print(f"  Action Verb Score:  {result.action_verb_score}")
    print(f"\nMatched Keywords: {result.matched_keywords[:5]}")
    print(f"Missing Keywords: {result.missing_keywords[:5]}")
    print(f"\nTop Improvement Tips:")
    for i, tip in enumerate(result.improvement_tips[:3], 1):
        print(f"  {i}. [{tip['priority']}] {tip['category']}: {tip['recommendation']}")
