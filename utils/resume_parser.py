"""
utils/resume_parser.py
────────────────────────────────────────────────────────────────────────────────
Handles PDF and DOCX resume parsing using PyMuPDF and python-docx.
Extracts structured fields: name, contact, skills, education, experience, etc.
────────────────────────────────────────────────────────────────────────────────
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

# Graceful imports — fall back if heavy deps not installed
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ── Skill master list (subset) ────────────────────────────────────────────────
KNOWN_SKILLS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "kotlin",
    "swift", "go", "rust", "scala", "php", "ruby",
    # Web
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "html", "css", "tailwind", "bootstrap", "nextjs",
    # Data / ML
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "xgboost", "lightgbm", "nlp", "computer vision", "bert", "transformers",
    "huggingface", "faiss", "spark", "hadoop",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "sqlite",
    "oracle", "elasticsearch",
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "linux", "bash",
    # Tools
    "git", "jira", "confluence", "tableau", "power bi", "excel", "postman",
    "swagger", "graphql", "rest api", "kafka", "rabbitmq",
    # Soft / Domain
    "agile", "scrum", "communication", "leadership", "teamwork",
    "problem solving", "time management",
]

SECTION_HEADERS = {
    "summary": ["summary", "objective", "profile", "about"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "experience": ["experience", "work experience", "employment", "work history"],
    "education": ["education", "academic", "qualification"],
    "projects": ["projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates", "credentials", "awards"],
    "contact": ["contact", "personal information", "details"],
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{10,15}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)


@dataclass
class ParsedResume:
    raw_text: str = ""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    skills: list = field(default_factory=list)
    education: list = field(default_factory=list)
    experience: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    summary: Optional[str] = None
    sections: dict = field(default_factory=dict)
    word_count: int = 0
    has_action_verbs: bool = False

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "linkedin": self.linkedin,
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,
            "projects": self.projects,
            "certifications": self.certifications,
            "summary": self.summary,
            "word_count": self.word_count,
            "has_action_verbs": self.has_action_verbs,
        }


ACTION_VERBS = [
    "developed", "designed", "implemented", "built", "optimized", "led",
    "managed", "delivered", "achieved", "reduced", "improved", "increased",
    "automated", "architected", "launched", "created", "deployed", "analysed",
    "coordinated", "established", "executed", "generated", "integrated",
]


class ResumeParser:
    """
    Parses PDF or DOCX resumes and extracts structured fields.
    Accepts raw text as well (for pipeline use).
    """

    def parse_text(self, text: str) -> ParsedResume:
        resume = ParsedResume()
        resume.raw_text = text
        resume.word_count = len(text.split())

        # Contact extraction
        emails = EMAIL_RE.findall(text)
        if emails:
            resume.email = emails[0]

        phones = PHONE_RE.findall(text)
        if phones:
            resume.phone = phones[0].strip()

        linkedin_matches = LINKEDIN_RE.findall(text)
        if linkedin_matches:
            resume.linkedin = linkedin_matches[0]

        # Name heuristic: first non-empty line that's not an email/url
        for line in text.strip().split("\n"):
            line = line.strip()
            if (line and len(line.split()) <= 5
                    and not EMAIL_RE.search(line)
                    and not URL_RE.search(line)
                    and not any(h in line.lower() for hlist in SECTION_HEADERS.values() for h in hlist)):
                resume.name = line
                break

        # Skills extraction
        text_lower = text.lower()
        resume.skills = [s for s in KNOWN_SKILLS if s in text_lower]

        # Action verbs check
        resume.has_action_verbs = any(v in text_lower for v in ACTION_VERBS)

        # Section-wise extraction
        resume.sections = self._extract_sections(text)

        resume.summary = resume.sections.get("summary", "")
        resume.experience = self._parse_bullets(resume.sections.get("experience", ""))
        resume.education = self._parse_bullets(resume.sections.get("education", ""))
        resume.projects = self._parse_bullets(resume.sections.get("projects", ""))
        resume.certifications = self._parse_bullets(resume.sections.get("certifications", ""))

        return resume

    def parse_pdf(self, file_path: str) -> ParsedResume:
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return self.parse_text(text)

    def parse_docx(self, file_path: str) -> ParsedResume:
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return self.parse_text(text)

    def parse_file(self, file_path: str) -> ParsedResume:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.parse_pdf(file_path)
        elif ext == ".docx":
            return self.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")

    def _extract_sections(self, text: str) -> dict:
        lines = text.split("\n")
        sections = {}
        current_section = "header"
        buffer = []

        for line in lines:
            stripped = line.strip()
            matched_section = self._match_section_header(stripped)
            if matched_section:
                if buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = matched_section
                buffer = []
            else:
                if stripped:
                    buffer.append(stripped)

        if buffer:
            sections[current_section] = "\n".join(buffer).strip()

        return sections

    def _match_section_header(self, line: str) -> Optional[str]:
        line_lower = line.lower().strip(":").strip()
        for section, keywords in SECTION_HEADERS.items():
            if any(kw in line_lower for kw in keywords):
                if len(line_lower) < 40:  # headers are short lines
                    return section
        return None

    def _parse_bullets(self, text: str) -> list:
        if not text:
            return []
        lines = [l.strip().lstrip("•-*").strip() for l in text.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5]


if __name__ == "__main__":
    parser = ResumeParser()

    sample_resume = """
John Doe
john.doe@email.com | +91-9876543210 | linkedin.com/in/johndoe

SUMMARY
Experienced Data Scientist with 4 years of experience in Machine Learning and Python.
Delivered multiple NLP projects reducing processing time by 40%.

SKILLS
Python, Machine Learning, TensorFlow, PyTorch, SQL, Pandas, NumPy, Scikit-learn, AWS, Docker

EXPERIENCE
• Developed end-to-end ML pipeline using Python and TensorFlow at Google.
• Optimized NLP models achieving 92% accuracy on classification tasks.
• Designed RESTful APIs using FastAPI and Docker.

EDUCATION
B.Tech in Computer Science, IIT Bombay, 2020

CERTIFICATIONS
AWS Certified Solutions Architect
TensorFlow Developer Certificate

PROJECTS
• Resume Screening System: Built NLP-based ATS scoring tool using BERT and FAISS.
"""

    result = parser.parse_text(sample_resume)
    print("Name:", result.name)
    print("Email:", result.email)
    print("Skills:", result.skills[:8])
    print("Word count:", result.word_count)
    print("Has action verbs:", result.has_action_verbs)
    print("Sections found:", list(result.sections.keys()))
    print("Experience bullets:", result.experience[:2])
