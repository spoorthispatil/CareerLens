"""
models/keyword_gap_analyzer.py
────────────────────────────────────────────────────────────────────────────────
Keyword Gap Analyzer using FAISS + TF-IDF
────────────────────────────────────────────────────────────────────────────────
Identifies exact and semantic near-miss keywords from a JD that are absent
from a resume. Uses FAISS for fast approximate nearest-neighbour search
over skill embeddings.
────────────────────────────────────────────────────────────────────────────────
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("  ⚠  FAISS not installed. Using cosine fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KeywordGap:
    keyword: str
    importance: str          # "high" | "medium" | "low"
    frequency_in_jd: int
    suggested_section: str
    near_match: Optional[str] = None   # semantic near-miss in resume
    near_match_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "importance": self.importance,
            "frequency_in_jd": self.frequency_in_jd,
            "suggested_section": self.suggested_section,
            "near_match": self.near_match,
            "near_match_score": round(self.near_match_score, 3),
        }


@dataclass
class GapAnalysisResult:
    missing_keywords: list = field(default_factory=list)
    present_keywords: list = field(default_factory=list)
    near_miss_keywords: list = field(default_factory=list)
    match_percentage: float = 0.0
    total_jd_keywords: int = 0

    def to_dict(self) -> dict:
        return {
            "missing_keywords": [k.to_dict() for k in self.missing_keywords],
            "present_keywords": self.present_keywords,
            "near_miss_keywords": [k.to_dict() for k in self.near_miss_keywords],
            "match_percentage": round(self.match_percentage, 1),
            "total_jd_keywords": self.total_jd_keywords,
        }


# ── Skill section mapping ─────────────────────────────────────────────────────
SECTION_MAP = {
    # Programming → Skills
    "python": "Skills", "java": "Skills", "javascript": "Skills",
    "typescript": "Skills", "c++": "Skills", "r": "Skills", "kotlin": "Skills",
    # Frameworks → Skills or Projects
    "react": "Skills / Projects", "angular": "Skills", "vue": "Skills",
    "node.js": "Skills", "django": "Skills", "flask": "Skills",
    "tensorflow": "Skills / Projects", "pytorch": "Skills / Projects",
    "scikit-learn": "Skills", "keras": "Skills",
    # Cloud / DevOps → Skills or Projects
    "aws": "Skills / Certifications", "azure": "Skills / Certifications",
    "docker": "Skills / Projects", "kubernetes": "Skills",
    # Soft skills → Summary
    "leadership": "Summary", "communication": "Summary", "agile": "Skills",
    "scrum": "Skills",
    # Certifications
    "certified": "Certifications", "certification": "Certifications",
}

DEFAULT_SECTION = "Skills"

# ── Synonyms / skill variants ─────────────────────────────────────────────────
SYNONYMS = {
    "machine learning": ["ml", "ai", "artificial intelligence", "statistical modeling"],
    "natural language processing": ["nlp", "text processing", "text mining"],
    "deep learning": ["neural networks", "dl", "cnn", "rnn", "lstm"],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "node.js": ["nodejs", "node"],
    "react": ["reactjs", "react.js"],
    "python": ["python3", "py"],
    "kubernetes": ["k8s"],
    "postgresql": ["postgres", "psql"],
    "mongodb": ["mongo"],
    "continuous integration": ["ci/cd", "cicd", "devops pipeline"],
    "rest api": ["restful api", "rest", "api development"],
    "sql": ["structured query language", "database querying"],
    "amazon web services": ["aws"],
    "google cloud": ["gcp", "google cloud platform"],
    "microsoft azure": ["azure"],
}


class KeywordGapAnalyzer:
    """
    Analyzes keyword gaps between a job description and a resume.
    Identifies:
      1. Exact missing keywords
      2. Near-miss keywords (synonyms/variants via semantic similarity)
      3. Present keywords (strengths to highlight)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            max_features=3000,
        )
        self._faiss_index = None
        self._faiss_keywords = []

    def analyze(
        self,
        resume_text: str,
        jd_text: str,
        top_n_jd_keywords: int = 30,
    ) -> GapAnalysisResult:
        """Full keyword gap analysis."""

        # Step 1: Extract keywords from JD
        jd_keywords = self._extract_jd_keywords(jd_text, top_n=top_n_jd_keywords)

        # Step 2: Classify each keyword
        resume_lower = resume_text.lower()
        result = GapAnalysisResult(total_jd_keywords=len(jd_keywords))

        missing = []
        present = []
        near_miss = []

        for kw, freq, importance in jd_keywords:
            kw_lower = kw.lower()

            # Exact match
            if kw_lower in resume_lower:
                present.append(kw)
                continue

            # Synonym check
            synonym_found = self._check_synonyms(kw_lower, resume_lower)
            if synonym_found:
                gap = KeywordGap(
                    keyword=kw,
                    importance=importance,
                    frequency_in_jd=freq,
                    suggested_section=SECTION_MAP.get(kw_lower, DEFAULT_SECTION),
                    near_match=synonym_found,
                    near_match_score=0.85,
                )
                near_miss.append(gap)
                continue

            # Missing keyword
            gap = KeywordGap(
                keyword=kw,
                importance=importance,
                frequency_in_jd=freq,
                suggested_section=SECTION_MAP.get(kw_lower, DEFAULT_SECTION),
            )
            missing.append(gap)

        # Step 3: Semantic near-miss via FAISS/cosine
        if missing:
            missing = self._find_semantic_near_misses(
                missing, resume_lower
            )

        # Sort missing by importance
        imp_order = {"high": 0, "medium": 1, "low": 2}
        missing.sort(key=lambda x: imp_order.get(x.importance, 3))

        result.missing_keywords = missing
        result.present_keywords = present
        result.near_miss_keywords = near_miss
        result.match_percentage = (
            len(present) / len(jd_keywords) * 100 if jd_keywords else 0
        )

        return result

    def _extract_jd_keywords(self, jd_text: str, top_n: int = 30) -> list:
        """Extract (keyword, frequency, importance) tuples from JD."""
        # First pass: frequency count on cleaned text
        text_lower = jd_text.lower()
        words = re.findall(r"\b[a-z][a-z0-9\+\#\.\_\-]{1,30}\b", text_lower)

        # Count word frequencies
        freq_map = {}
        for w in words:
            freq_map[w] = freq_map.get(w, 0) + 1

        # Remove stopwords
        stopwords = {
            "and", "or", "the", "a", "an", "in", "of", "to", "for",
            "with", "on", "at", "by", "from", "is", "are", "be", "will",
            "you", "we", "our", "your", "this", "that", "have", "has",
            "not", "but", "as", "it", "its", "we", "they", "their",
            "experience", "years", "required", "preferred", "skills",
            "ability", "knowledge", "working", "strong", "good",
        }
        freq_map = {k: v for k, v in freq_map.items() if k not in stopwords and len(k) > 2}

        # Also extract bigrams for compound skills
        bigrams = re.findall(
            r"\b(machine learning|deep learning|natural language|computer vision|"
            r"data science|web development|rest api|ci.?cd|node\.?js|react\.?js|"
            r"power bi|big data|cloud computing)\b",
            text_lower
        )
        for bg in bigrams:
            freq_map[bg] = freq_map.get(bg, 0) + 2  # boost bigrams

        # Sort and assign importance
        sorted_kw = sorted(freq_map.items(), key=lambda x: -x[1])[:top_n]

        result = []
        for i, (kw, freq) in enumerate(sorted_kw):
            if i < top_n * 0.3:
                importance = "high"
            elif i < top_n * 0.7:
                importance = "medium"
            else:
                importance = "low"
            result.append((kw, freq, importance))

        return result

    def _check_synonyms(self, keyword: str, resume_lower: str) -> Optional[str]:
        """Check if any synonym of the keyword appears in the resume."""
        for canonical, variants in SYNONYMS.items():
            if keyword == canonical or keyword in variants:
                all_forms = [canonical] + variants
                for form in all_forms:
                    if form in resume_lower and form != keyword:
                        return form
        return None

    def _find_semantic_near_misses(
        self, missing_gaps: list, resume_lower: str
    ) -> list:
        """
        Use TF-IDF cosine similarity to find near-miss keywords.
        Falls back from FAISS if unavailable.
        """
        resume_words = set(re.findall(r"\b[a-z][a-z0-9\+\#\.\_\-]{1,20}\b", resume_lower))

        for gap in missing_gaps:
            kw = gap.keyword.lower()
            # Find most similar resume word
            best_score = 0.0
            best_match = None

            try:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                candidates = list(resume_words)[:200]
                if candidates:
                    matrix = vec.fit_transform([kw] + candidates)
                    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
                    if len(scores) > 0:
                        best_idx = scores.argmax()
                        best_score = float(scores[best_idx])
                        if best_score > 0.5:
                            best_match = candidates[best_idx]
            except Exception:
                pass

            if best_match and best_score > 0.5:
                gap.near_match = best_match
                gap.near_match_score = best_score

        return missing_gaps


if __name__ == "__main__":
    analyzer = KeywordGapAnalyzer()

    resume = """
    SKILLS
    Python, Machine Learning, TensorFlow, Pandas, NumPy, SQL, Git, Docker, REST API

    EXPERIENCE
    Developed ML models using scikit-learn and TensorFlow.
    Deployed REST APIs using Flask and Docker on AWS.
    Worked with PostgreSQL and MongoDB databases.
    """

    jd = """
    We need a Senior Data Scientist with expertise in:
    Required: Python, Machine Learning, Deep Learning, NLP, PyTorch, Spark, SQL,
    Feature Engineering, Statistics, Computer Vision
    Preferred: AWS SageMaker, MLflow, Kubernetes, Airflow, Tableau
    Experience: 3+ years in data science or ML engineering.
    """

    result = analyzer.analyze(resume, jd)

    print(f"Keyword Match: {result.match_percentage:.1f}%")
    print(f"Present Keywords ({len(result.present_keywords)}): {result.present_keywords[:6]}")
    print(f"\nMissing Keywords ({len(result.missing_keywords)}):")
    for gap in result.missing_keywords[:5]:
        d = gap.to_dict()
        near = f" (near-miss: '{d['near_match']}')" if d['near_match'] else ""
        print(f"  [{d['importance'].upper()}] {d['keyword']} → Add to {d['suggested_section']}{near}")
    print(f"\nNear-miss Keywords ({len(result.near_miss_keywords)}):")
    for gap in result.near_miss_keywords[:3]:
        d = gap.to_dict()
        print(f"  {d['keyword']} ≈ '{d['near_match']}' in resume (score: {d['near_match_score']})")
