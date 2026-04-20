<div align="center">

# 🔍 CareerLens — AI Resume Intelligence

**An end-to-end ML system that scores, analyzes, and improves resumes using NLP and Transformer embeddings.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![BERT](https://img.shields.io/badge/BERT-all--MiniLM--L6--v2-yellow?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests](https://img.shields.io/badge/Tests-37%20passing-2ecc85?style=flat-square&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> *"Most resume analyzers count keywords. CareerLens understands meaning."*

<br/>

</div>

---

## 🎯 The Problem

Every year, **75% of resumes are rejected by ATS systems** before a human reads them — not because the candidate is unqualified, but because the resume doesn't use the right keywords, lacks structure, or fails formatting rules automated scanners enforce.

**CareerLens solves this** by giving every job seeker access to the same intelligence enterprise HR systems use — completely free.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **ATS Score Prediction** | Predicts ATS compatibility (0–100) via trained ML ensemble |
| **Semantic Keyword Matching** | Uses BERT (`all-MiniLM-L6-v2`) to find *meaning-level* matches, not just exact word overlap |
| **Keyword Gap Analysis** | Identifies missing JD keywords + near-miss synonyms via FAISS |
| **Resume Classifier** | Auto-detects job category across 8 domains (97%+ accuracy) |
| **Company-Specific Tips** | Tailored advice for 11 companies: Google, Amazon, TCS, Infosys and more |
| **Interview Prep** | Generates 12 targeted Q&A based on your resume + JD |
| **Score History** | Tracks every analysis so you can measure improvement over time |
| **Email Reports** | Beautiful HTML analysis reports via Gmail SMTP |
| **PDF / DOCX Upload** | Drag-and-drop file parsing with PyMuPDF |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CareerLens System                     │
│                                                          │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  Frontend    │    │        ML Pipeline           │   │
│  │  (HTML/JS/   │◄──►│                              │   │
│  │  Chart.js)   │    │  ┌──────────┐  ┌──────────┐  │   │
│  │              │    │  │  Resume  │  │   ATS    │  │   │
│  │  • Score     │    │  │  Parser  │  │  Scorer  │  │   │
│  │    Ring      │    │  │ PyMuPDF  │  │          │  │   │
│  │  • Radar     │    │  │  docx    │  │  TF-IDF  │  │   │
│  │  • Company   │    │  └────┬─────┘  │  + BERT  │  │   │
│  │    Fit Bars  │    │       │        │  + GBM   │  │   │
│  │  • Q&A Prep  │    │       ▼        └──────────┘  │   │
│  └──────────────┘    │  ┌──────────┐  ┌──────────┐  │   │
│                      │  │ Keyword  │  │ Category │  │   │
│  ┌──────────────┐    │  │  Gap     │  │Classifier│  │   │
│  │  FastAPI     │    │  │ Analyzer │  │ (LogReg) │  │   │
│  │  9 Endpoints │    │  │ (FAISS)  │  └──────────┘  │   │
│  └──────────────┘    │  └──────────┘                │   │
│                      │  ┌──────────────────────────┐ │   │
│  ┌──────────────┐    │  │  Company Recommender     │ │   │
│  │  Gmail SMTP  │    │  │  (11 company profiles)   │ │   │
│  │  HTML Email  │    │  └──────────────────────────┘ │   │
│  └──────────────┘    └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 🧠 How the ATS Scorer Works

```
Resume Text + Job Description
         │
         ├── Layer 1 ── TF-IDF Cosine Similarity
         │              (n-gram keyword overlap)
         │
         ├── Layer 2 ── Sentence-BERT (all-MiniLM-L6-v2)
         │              • Full document semantic similarity
         │              • Per-skill semantic matching
         │              • Section-weighted scoring (experience > summary)
         │
         ├── Layer 3 ── Heuristic Scorer
         │              • Formatting  (word count, bullets, quantification)
         │              • Completeness (sections present)
         │              • Readability  (sentence length, vocab diversity)
         │              • Action verb density
         │
         └── Layer 4 ── Gradient Boosted Meta-Learner
                        Trained on 600 labeled resumes
                                 │
                          ATS Score (0–100)
                          + Grade + Tips + Matched/Missing Keywords
```

---

## 📁 Project Structure

```
careerlens/
├── train_pipeline.py          # Main ML training script
├── requirements.txt
├── .env.example               # Gmail SMTP config template
│
├── data/
│   ├── generate_dataset.py    # Dataset generator (mirrors Kaggle UpdatedResumeDataSet)
│   └── resumes.csv
│
├── models/
│   ├── ats_scorer.py          # ⭐ Core ATS engine — BERT + 4-layer ensemble
│   ├── resume_classifier.py   # TF-IDF + Logistic Regression classifier
│   ├── keyword_gap_analyzer.py # FAISS + BERT keyword gap detection
│   └── company_recommender.py  # 11-company tip engine
│
├── utils/
│   ├── resume_parser.py       # PDF/DOCX parser (PyMuPDF + python-docx)
│   └── email_service.py       # Gmail SMTP HTML email service
│
├── api/
│   └── app.py                 # FastAPI REST API — 9 endpoints
│
├── frontend/
│   └── index.html             # Dashboard (Vanilla JS + Chart.js)
│
├── notebooks/
│   └── resume_screening_eda.ipynb
│
└── tests/
    └── test_pipeline.py       # 37 unit + integration tests
```

---

## 🗂 Dataset

**Source:** [Kaggle — UpdatedResumeDataSet](https://www.kaggle.com/datasets/dhainjeamita/updatedresumedataset)
- **2,484 real resumes** across 25 job categories
- Consolidated into 8 primary categories for training

**To use it:**
```
1. Download from Kaggle (link above)
2. Place as:  data/UpdatedResumeDataSet.csv
3. Run:       python train_pipeline.py
```
The pipeline auto-detects the file. If not present, it falls back to a 600-resume synthetic dataset so the project always works out of the box.

---



```bash
# 1. Clone
git clone https://github.com/spoorthispatil/careerlens.git
cd careerlens

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install BERT for semantic scoring
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
# → App auto-upgrades from TF-IDF to BERT. You'll see ⚡ BERT Active in navbar.

# 4. (Recommended) Use the real Kaggle dataset — 2,484 real resumes
#    a. Download: https://www.kaggle.com/datasets/dhainjeamita/updatedresumedataset
#    b. Place the CSV as:  data/UpdatedResumeDataSet.csv
#    c. The training pipeline auto-detects it. No other changes needed.

# 5. Train ML models (~30 seconds on real data)
python train_pipeline.py

# 6. Launch
python -m uvicorn api.app:app --reload --port 8000
```

Open **http://127.0.0.1:8000** — full dashboard loads immediately.

---

## 📧 Enable Email Reports

```bash
# 1. Enable 2FA on Gmail
# 2. Go to: myaccount.google.com/apppasswords → Create App Password for "Mail"
# 3. Create your .env file:
cp .env.example .env

# Fill in:
GMAIL_SENDER=your.email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/health` | Health + BERT status |
| `POST` | `/score` | ATS score prediction |
| `POST` | `/classify` | Category classifier |
| `POST` | `/gap-analysis` | Keyword gap analysis |
| `GET` | `/companies` | List company profiles |
| `POST` | `/company-tips` | Company-specific tips |
| `POST` | `/full-analysis` | All-in-one endpoint |
| `POST` | `/send-report` | Email analysis report |
| `POST` | `/upload-resume` | Parse PDF/DOCX |

Interactive docs → **http://127.0.0.1:8000/docs**

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| ATS Meta-Learner (GBM) | R² | 0.86 |
| Resume Classifier | Accuracy | 97.3% |
| BERT Semantic Threshold | Cosine | 0.45 |
| Keyword Coverage | % JD terms analyzed | ~92% |

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
# 37 tests — all passing
```

---

## 🔭 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn · GradientBoosting · LogisticRegression |
| NLP | Sentence-BERT `all-MiniLM-L6-v2` · TF-IDF |
| Semantic Search | FAISS · cosine similarity |
| Resume Parsing | PyMuPDF · python-docx |
| API | FastAPI · Uvicorn · Pydantic |
| Frontend | Vanilla JS · Chart.js |
| Email | Gmail SMTP · smtplib · HTML MIME |
| Testing | pytest · 37 tests |

---

## 🗺 Roadmap

- [ ] Deploy to Railway (live demo URL)
- [ ] Fine-tune on real Kaggle resume dataset
- [ ] PDF export of analysis report
- [ ] Resume rewrite suggestions via LLM

---

## 👩‍💻 Author

**Spoorthi S Patil** — B.E. Computer Science

[![GitHub](https://img.shields.io/badge/GitHub-spoorthispatil-181717?style=flat-square&logo=github)](https://github.com/spoorthispatil)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/spoorthispatil)

---

<div align="center">
  <sub>Built with Python, ML, and a lot of ☕ · Portfolio Project 2025</sub>
</div>
