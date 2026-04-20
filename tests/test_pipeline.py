"""
tests/test_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for all ML components.
Run: python -m pytest tests/ -v
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.resume_parser import ResumeParser
from models.ats_scorer import ATSScorer, HeuristicScorer, TFIDFScorer
from models.keyword_gap_analyzer import KeywordGapAnalyzer
from models.company_recommender import CompanyRecommender

# ── Fixtures ─────────────────────────────────────────────────────────────────
SAMPLE_RESUME = """
John Doe | john@email.com | +91-9876543210 | linkedin.com/in/johndoe

SUMMARY
Data Scientist with 3 years of experience in Python and Machine Learning.

SKILLS
Python, Machine Learning, TensorFlow, SQL, Pandas, Docker, AWS, REST API

EXPERIENCE
• Developed ML pipeline using TensorFlow, reducing training time by 35%.
• Implemented NLP model achieving 91% accuracy.
• Deployed models on AWS serving 5,000 daily requests.

EDUCATION
B.Tech Computer Science, IIT Bombay, 2021

CERTIFICATIONS
AWS Certified Solutions Architect

PROJECTS
• Resume Screener: NLP ATS scoring using BERT.
"""

SAMPLE_JD = """
Data Scientist Required Skills: Python, Machine Learning, TensorFlow, NLP,
SQL, Feature Engineering, Scikit-learn, Statistics, Docker, AWS.
Preferred: PyTorch, Spark, Kubernetes, Tableau.
Experience: 2+ years in data science.
"""

WEAK_RESUME = """
John | john@gmail.com
Skills: Python
Experience: Worked on some projects.
Education: B.Tech 2023
"""


# ── Parser Tests ──────────────────────────────────────────────────────────────
class TestResumeParser:
    def setup_method(self):
        self.parser = ResumeParser()

    def test_parse_extracts_email(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert result.email == "john@email.com"

    def test_parse_extracts_phone(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert result.phone is not None
        assert "9876543210" in result.phone.replace("-", "").replace("+91", "")

    def test_parse_extracts_linkedin(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert result.linkedin is not None
        assert "linkedin" in result.linkedin

    def test_parse_extracts_skills(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert len(result.skills) > 0
        assert "python" in result.skills or "machine learning" in result.skills

    def test_parse_word_count(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert result.word_count > 50

    def test_parse_action_verbs(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert result.has_action_verbs is True

    def test_parse_sections(self):
        result = self.parser.parse_text(SAMPLE_RESUME)
        assert len(result.sections) >= 3

    def test_parse_empty_text(self):
        result = self.parser.parse_text("")
        assert result.skills == []
        assert result.word_count == 0

    def test_parse_weak_resume(self):
        result = self.parser.parse_text(WEAK_RESUME)
        assert result.email is not None


# ── ATS Scorer Tests ──────────────────────────────────────────────────────────
class TestATSScorer:
    def setup_method(self):
        self.scorer = ATSScorer()

    def test_score_returns_ats_score_object(self):
        from models.ats_scorer import ATSScore
        result = self.scorer.score(SAMPLE_RESUME)
        assert isinstance(result, ATSScore)

    def test_score_range_0_to_100(self):
        result = self.scorer.score(SAMPLE_RESUME)
        assert 0 <= result.overall_score <= 100

    def test_sub_scores_range(self):
        result = self.scorer.score(SAMPLE_RESUME)
        for attr in ["keyword_score", "formatting_score", "completeness_score",
                     "readability_score", "action_verb_score"]:
            score = getattr(result, attr)
            assert 0 <= score <= 100, f"{attr} out of range: {score}"

    def test_score_with_jd_higher_than_weak(self):
        strong_result = self.scorer.score(SAMPLE_RESUME, SAMPLE_JD)
        weak_result = self.scorer.score(WEAK_RESUME, SAMPLE_JD)
        assert strong_result.overall_score > weak_result.overall_score

    def test_improvement_tips_generated(self):
        result = self.scorer.score(WEAK_RESUME, SAMPLE_JD)
        assert len(result.improvement_tips) > 0

    def test_tips_have_required_fields(self):
        result = self.scorer.score(WEAK_RESUME, SAMPLE_JD)
        for tip in result.improvement_tips:
            assert "category" in tip
            assert "recommendation" in tip
            assert "priority" in tip
            assert tip["priority"] in ["High", "Medium", "Low"]

    def test_grade_classification(self):
        from models.ats_scorer import ATSScore
        import dataclasses
        # High score → Excellent
        high = ATSScore(85, 85, 85, 85, 85, 85, [], [], [], {})
        assert high.grade() == "Excellent"
        # Low score → Needs Improvement
        low = ATSScore(40, 40, 40, 40, 40, 40, [], [], [], {})
        assert low.grade() == "Needs Improvement"

    def test_score_to_dict(self):
        result = self.scorer.score(SAMPLE_RESUME)
        d = result.to_dict()
        assert "overall_score" in d
        assert "keyword_score" in d
        assert "improvement_tips" in d


# ── Heuristic Scorer Tests ────────────────────────────────────────────────────
class TestHeuristicScorer:
    def setup_method(self):
        self.scorer = HeuristicScorer()

    def test_formatting_good_resume(self):
        score = self.scorer.score_formatting(SAMPLE_RESUME)
        assert score > 50

    def test_formatting_weak_resume(self):
        score = self.scorer.score_formatting("Python developer.")
        assert score <= 70

    def test_completeness_full_resume(self):
        score = self.scorer.score_completeness(SAMPLE_RESUME)
        assert score >= 60

    def test_completeness_missing_sections(self):
        score = self.scorer.score_completeness("Just some random text.")
        assert score < 30

    def test_action_verbs_strong_resume(self):
        score = self.scorer.score_action_verbs(SAMPLE_RESUME)
        assert score > 50

    def test_readability_returns_in_range(self):
        score = self.scorer.score_readability(SAMPLE_RESUME)
        assert 0 <= score <= 100


# ── Keyword Gap Analyzer Tests ────────────────────────────────────────────────
class TestKeywordGapAnalyzer:
    def setup_method(self):
        self.analyzer = KeywordGapAnalyzer()

    def test_analyze_returns_result(self):
        from models.keyword_gap_analyzer import GapAnalysisResult
        result = self.analyzer.analyze(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result, GapAnalysisResult)

    def test_match_percentage_range(self):
        result = self.analyzer.analyze(SAMPLE_RESUME, SAMPLE_JD)
        assert 0 <= result.match_percentage <= 100

    def test_good_resume_higher_match(self):
        good = self.analyzer.analyze(SAMPLE_RESUME, SAMPLE_JD)
        weak = self.analyzer.analyze(WEAK_RESUME, SAMPLE_JD)
        assert good.match_percentage >= weak.match_percentage

    def test_missing_keywords_have_importance(self):
        result = self.analyzer.analyze(WEAK_RESUME, SAMPLE_JD)
        for gap in result.missing_keywords:
            assert gap.importance in ["high", "medium", "low"]

    def test_to_dict_structure(self):
        result = self.analyzer.analyze(SAMPLE_RESUME, SAMPLE_JD)
        d = result.to_dict()
        assert "missing_keywords" in d
        assert "present_keywords" in d
        assert "match_percentage" in d


# ── Company Recommender Tests ─────────────────────────────────────────────────
class TestCompanyRecommender:
    def setup_method(self):
        self.recommender = CompanyRecommender()

    def test_list_companies_not_empty(self):
        companies = self.recommender.list_companies()
        assert len(companies) >= 5

    def test_google_in_companies(self):
        companies = self.recommender.list_companies()
        assert "Google" in companies

    def test_get_tips_returns_list(self):
        tips = self.recommender.get_tips("Google", SAMPLE_RESUME)
        assert isinstance(tips, list)
        assert len(tips) > 0

    def test_tips_have_required_fields(self):
        tips = self.recommender.get_tips("Amazon", SAMPLE_RESUME)
        for tip in tips:
            assert "company" in tip
            assert "message" in tip
            assert "priority" in tip
            assert "tip_type" in tip

    def test_unknown_company_returns_fallback(self):
        tips = self.recommender.get_tips("UnknownXYZ", SAMPLE_RESUME)
        assert len(tips) > 0

    def test_compare_companies(self):
        fit = self.recommender.compare_companies(SAMPLE_RESUME, ["Google", "TCS", "Amazon"])
        assert len(fit) == 3
        for company, data in fit.items():
            assert "fit_score" in data
            assert 0 <= data["fit_score"] <= 100

    def test_compare_companies_sorted_descending(self):
        fit = self.recommender.compare_companies(SAMPLE_RESUME, ["Google", "TCS", "Amazon"])
        scores = [v["fit_score"] for v in fit.values()]
        assert scores == sorted(scores, reverse=True)


# ── Integration Test ──────────────────────────────────────────────────────────
class TestIntegration:
    """End-to-end pipeline test."""

    def test_full_pipeline_strong_resume(self):
        parser = ResumeParser()
        scorer = ATSScorer()
        analyzer = KeywordGapAnalyzer()
        recommender = CompanyRecommender()

        parsed = parser.parse_text(SAMPLE_RESUME)
        assert parsed.email is not None

        ats = scorer.score(parsed.raw_text, SAMPLE_JD)
        assert ats.overall_score > 30

        gap = analyzer.analyze(parsed.raw_text, SAMPLE_JD)
        assert gap.match_percentage >= 0

        tips = recommender.get_tips("Google", parsed.raw_text)
        assert len(tips) > 0

    def test_full_pipeline_weak_resume_produces_tips(self):
        parser = ResumeParser()
        scorer = ATSScorer()

        parsed = parser.parse_text(WEAK_RESUME)
        ats = scorer.score(parsed.raw_text, SAMPLE_JD)

        assert ats.overall_score < 70
        assert len(ats.improvement_tips) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
