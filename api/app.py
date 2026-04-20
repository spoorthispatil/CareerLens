"""
api/app.py
────────────────────────────────────────────────────────────────────────────────
FastAPI ML Microservice — Resume Screening System
────────────────────────────────────────────────────────────────────────────────
Exposes endpoints:
  POST /score           → ATS score a resume against optional JD
  POST /classify        → Predict resume category
  POST /gap-analysis    → Keyword gap analysis
  GET  /companies       → List available company profiles
  POST /company-tips    → Company-specific resume tips
  POST /full-analysis   → All-in-one analysis endpoint
  GET  /health          → Health check

Usage:
  uvicorn api.app:app --reload --port 8000
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models.ats_scorer import ATSScorer
from models.resume_classifier import ResumeClassifier
from models.keyword_gap_analyzer import KeywordGapAnalyzer
from models.company_recommender import CompanyRecommender
from utils.resume_parser import ResumeParser
from utils.email_service import send_report as send_email_report_util, is_configured as email_is_configured

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CareerLens — AI Resume Intelligence API",
    description="AI-Powered ATS Score Prediction & Resume Enhancement by Spoorthi S Patil",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────────────────────
ats_scorer = ATSScorer()
classifier = ResumeClassifier()
gap_analyzer = KeywordGapAnalyzer()
company_recommender = CompanyRecommender()
parser = ResumeParser()

try:
    ats_scorer.load_model()
except Exception:
    pass

try:
    classifier.load_model()
except Exception:
    pass


# ── Request / Response Models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    resume_text: str
    jd_text: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Python Developer with 3 years experience...",
                "jd_text": "We need a Python Developer with Django, REST API..."
            }
        }


class ClassifyRequest(BaseModel):
    resume_text: str


class GapAnalysisRequest(BaseModel):
    resume_text: str
    jd_text: str


class CompanyTipsRequest(BaseModel):
    resume_text: str
    company: str


class FullAnalysisRequest(BaseModel):
    resume_text: str
    jd_text: Optional[str] = None
    company: Optional[str] = None


class EmailReportRequest(BaseModel):
    recipient_email: str
    recipient_name: Optional[str] = ""
    analysis: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CareerLens — AI Resume Intelligence",
        "version": "2.0.0",
        "author": "Spoorthi S Patil",
        "bert_active": ats_scorer.semantic_scorer.is_bert_active,
        "timestamp": time.time(),
    }

@app.get("/")
def root():
    """Serve the CareerLens frontend dashboard."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "CareerLens API running. Visit /docs for API documentation."}


@app.post("/score")
def score_resume(request: ScoreRequest):
    """
    Predict ATS score for a resume.
    Returns overall score (0-100), sub-scores, and improvement tips.
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
    if len(request.resume_text) < 50:
        raise HTTPException(status_code=400, detail="Resume text too short (min 50 chars)")

    t0 = time.time()
    result = ats_scorer.score(request.resume_text, request.jd_text)
    elapsed = round((time.time() - t0) * 1000, 1)

    return {
        "overall_score": result.overall_score,
        "grade": result.grade(),
        "sub_scores": {
            "keyword_score": result.keyword_score,
            "formatting_score": result.formatting_score,
            "completeness_score": result.completeness_score,
            "readability_score": result.readability_score,
            "action_verb_score": result.action_verb_score,
        },
        "matched_keywords": result.matched_keywords[:10],
        "missing_keywords": result.missing_keywords[:10],
        "improvement_tips": result.improvement_tips,
        "processing_time_ms": elapsed,
    }


@app.post("/classify")
def classify_resume(request: ClassifyRequest):
    """Predict the job category of a resume."""
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")

    try:
        result = classifier.predict(request.resume_text)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not trained. Run train_pipeline.py first."
        )


@app.post("/gap-analysis")
def gap_analysis(request: GapAnalysisRequest):
    """Analyze keyword gaps between resume and job description."""
    if not request.resume_text.strip() or not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Both resume_text and jd_text required")

    result = gap_analyzer.analyze(request.resume_text, request.jd_text)
    return result.to_dict()


@app.get("/companies")
def list_companies():
    """List all available company profiles."""
    return {
        "companies": company_recommender.list_companies(),
        "total": len(company_recommender.list_companies())
    }


@app.post("/company-tips")
def company_tips(request: CompanyTipsRequest):
    """Get company-specific resume improvement tips."""
    companies = company_recommender.list_companies()
    if request.company not in companies:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{request.company}' not found. Available: {companies}"
        )

    tips = company_recommender.get_tips(request.company, request.resume_text)
    return {
        "company": request.company,
        "tips": tips,
        "total_tips": len(tips),
    }


@app.post("/full-analysis")
def full_analysis(request: FullAnalysisRequest):
    """
    All-in-one endpoint: ATS score + classification + gap analysis + company tips.
    This is the main endpoint for the frontend dashboard.
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")

    t0 = time.time()

    # Parse resume structure
    parsed = parser.parse_text(request.resume_text)

    # ATS Score
    ats_result = ats_scorer.score(request.resume_text, request.jd_text)

    # Category classification
    try:
        category_result = classifier.predict(request.resume_text)
    except Exception:
        category_result = {"predicted_category": "Unknown", "confidence": 0, "top_3": []}

    # Keyword gap analysis (only if JD provided)
    gap_result = None
    if request.jd_text:
        gap_result = gap_analyzer.analyze(request.resume_text, request.jd_text).to_dict()

    # Company tips (only if company provided)
    company_tips_result = None
    if request.company:
        company_tips_result = company_recommender.get_tips(request.company, request.resume_text)

    # Company fit comparison
    all_companies = company_recommender.list_companies()
    company_fit = company_recommender.compare_companies(request.resume_text, all_companies[:8])

    elapsed = round((time.time() - t0) * 1000, 1)

    return {
        "parsed_info": {
            "name": parsed.name,
            "email": parsed.email,
            "phone": parsed.phone,
            "linkedin": parsed.linkedin,
            "skills": parsed.skills[:15],
            "word_count": parsed.word_count,
            "sections_found": list(parsed.sections.keys()),
        },
        "ats_score": {
            "overall": ats_result.overall_score,
            "grade": ats_result.grade(),
            "sub_scores": {
                "keyword": ats_result.keyword_score,
                "formatting": ats_result.formatting_score,
                "completeness": ats_result.completeness_score,
                "readability": ats_result.readability_score,
                "action_verbs": ats_result.action_verb_score,
            },
            "matched_keywords": ats_result.matched_keywords[:10],
            "missing_keywords": ats_result.missing_keywords[:10],
        },
        "improvement_tips": ats_result.improvement_tips,
        "category": category_result,
        "keyword_gap": gap_result,
        "company_tips": company_tips_result,
        "company_fit_ranking": company_fit,
        "processing_time_ms": elapsed,
    }


@app.post("/send-report")
def send_report(request: EmailReportRequest):
    """
    Send the CareerLens analysis report to an email address via Gmail SMTP.
    Requires GMAIL_SENDER and GMAIL_APP_PASSWORD in .env file.
    """
    result = send_email_report_util(
        recipient_email=request.recipient_email,
        analysis=request.analysis,
        recipient_name=request.recipient_name or "",
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@app.get("/email-status")
def email_status():
    """Check if Gmail SMTP is configured."""
    return {
        "configured": email_is_configured(),
        "message": "Gmail SMTP ready." if email_is_configured()
                   else "Not configured. Add GMAIL_SENDER and GMAIL_APP_PASSWORD to .env"
    }


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX resume file and extract text.
    Returns parsed resume text for downstream scoring.
    """
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or DOCX."
        )

    max_size = 5 * 1024 * 1024  # 5MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

    # Save temp file
    import tempfile
    suffix = ".pdf" if "pdf" in file.content_type else ".docx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parsed = parser.parse_file(tmp_path)
        return {
            "filename": file.filename,
            "file_size_kb": round(len(content) / 1024, 1),
            "extracted_text": parsed.raw_text,
            "word_count": parsed.word_count,
            "parsed_fields": {
                "name": parsed.name,
                "email": parsed.email,
                "skills": parsed.skills[:10],
                "sections": list(parsed.sections.keys()),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse resume: {str(e)}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
