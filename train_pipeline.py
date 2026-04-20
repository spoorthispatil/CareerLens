"""
train_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Main Training Pipeline for CodeBlaze Resume Screening ML System
────────────────────────────────────────────────────────────────────────────────
Runs the full ML pipeline:
  1. Dataset generation (or load from CSV)
  2. Resume parsing & feature extraction
  3. ATS Scorer meta-learner training
  4. Resume Category Classifier training
  5. Model evaluation & reporting
  6. Save all artifacts

Usage:
  python train_pipeline.py
  python train_pipeline.py --data data/resumes.csv
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data.generate_dataset import generate_resumes, generate_job_descriptions, generate_company_profiles
from models.ats_scorer import ATSScorer, HeuristicScorer
from models.resume_classifier import ResumeClassifier
from models.keyword_gap_analyzer import KeywordGapAnalyzer
from models.company_recommender import CompanyRecommender
from utils.resume_parser import ResumeParser

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def banner(msg: str):
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")


def step1_generate_data(data_path: str = None) -> pd.DataFrame:
    banner("STEP 1 — Dataset Preparation")

    # Priority 1: explicit path argument
    if data_path and os.path.exists(data_path):
        print(f"  Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)

    # Priority 2: real Kaggle CSV dropped in data/
    elif os.path.exists("data/UpdatedResumeDataSet.csv"):
        print("  ✅ Real Kaggle dataset detected! (UpdatedResumeDataSet.csv)")
        print("  Processing with load_real_dataset.py...")
        from data.load_real_dataset import load_and_prepare
        df = load_and_prepare("data/UpdatedResumeDataSet.csv")
        if df is None:
            print("  ⚠  Failed to load real dataset. Falling back to synthetic.")
            df = generate_resumes(600)
            df.to_csv("data/resumes.csv", index=False)

    # Priority 3: pre-existing processed CSV
    elif os.path.exists("data/resumes.csv"):
        print("  Loading existing resumes.csv...")
        df = pd.read_csv("data/resumes.csv")

    # Priority 4: generate synthetic
    else:
        print("  Generating synthetic dataset (mirrors Kaggle UpdatedResumeDataSet)...")
        print("  💡 Tip: Download the real dataset for better model performance!")
        print("     https://www.kaggle.com/datasets/dhainjeamita/updatedresumedataset")
        print("     Place as:  data/UpdatedResumeDataSet.csv  then re-run.")
        df = generate_resumes(600)
        df.to_csv("data/resumes.csv", index=False)
        print(f"  ✓ Generated {len(df)} synthetic resumes")

    print(f"\n  Dataset: {len(df)} resumes × {df.shape[1]} columns")
    print(f"  Source:  {'REAL Kaggle data ✅' if os.path.exists('data/UpdatedResumeDataSet.csv') else 'Synthetic'}")
    print(f"  Category distribution:\n{df['category'].value_counts().to_string()}")
    return df


def step2_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2 — Feature Extraction")
    heuristic = HeuristicScorer()

    print("  Extracting heuristic sub-scores from resumes...")
    formatting_scores = []
    completeness_scores = []
    readability_scores = []
    action_verb_scores = []

    for text in tqdm(df["resume_text"], desc="  Extracting features"):
        formatting_scores.append(heuristic.score_formatting(text))
        completeness_scores.append(heuristic.score_completeness(text))
        readability_scores.append(heuristic.score_readability(text))
        action_verb_scores.append(heuristic.score_action_verbs(text))

    df["formatting_score"] = formatting_scores
    df["completeness_score"] = completeness_scores
    df["readability_score"] = readability_scores
    df["action_verb_score"] = action_verb_scores

    print(f"  ✓ Features extracted for {len(df)} resumes")
    print(f"  Feature stats:\n{df[['formatting_score','completeness_score','readability_score','action_verb_score']].describe().round(2).to_string()}")
    return df


def step3_train_ats_scorer(df: pd.DataFrame) -> tuple:
    banner("STEP 3 — ATS Scorer Training")
    scorer = ATSScorer()

    # Fit TF-IDF on resume corpus
    print("  Fitting TF-IDF vectorizer on resume corpus...")
    corpus = df["resume_text"].tolist()
    scorer.fit_tfidf(corpus)

    # Compute keyword scores using TF-IDF self-scoring
    print("  Computing keyword scores...")
    keyword_scores = []
    for text in tqdm(df["resume_text"], desc="  Keyword scoring"):
        score = scorer._general_keyword_density(text)
        keyword_scores.append(score)
    df["keyword_score"] = keyword_scores

    # Train meta-learner
    metrics = scorer.train_meta_model(df)
    return scorer, metrics


def step4_train_classifier(df: pd.DataFrame) -> tuple:
    banner("STEP 4 — Resume Category Classifier Training")
    classifier = ResumeClassifier()
    metrics = classifier.train(df)
    return classifier, metrics


def step5_evaluate_and_report(df, ats_scorer, classifier, ats_metrics, clf_metrics):
    banner("STEP 5 — Evaluation & Reporting")

    report = {
        "ats_scorer": ats_metrics,
        "classifier": {
            "cv_accuracy": clf_metrics["cv_accuracy"],
            "test_accuracy": clf_metrics["test_accuracy"],
            "classes": clf_metrics["classes"],
        },
        "dataset": {
            "total_resumes": len(df),
            "categories": df["category"].value_counts().to_dict(),
            "ats_score_distribution": {
                "mean": float(df["ats_score"].mean()),
                "std": float(df["ats_score"].std()),
                "min": float(df["ats_score"].min()),
                "max": float(df["ats_score"].max()),
            }
        }
    }

    # Save JSON report
    report_path = os.path.join(REPORTS_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Training report saved: {report_path}")

    # ── Plot 1: ATS Score Distribution ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CodeBlaze Resume Screening — Training Dashboard", fontsize=16, fontweight="bold")

    axes[0, 0].hist(df["ats_score"], bins=30, color="#4F8EF7", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("ATS Score Distribution")
    axes[0, 0].set_xlabel("ATS Score (0-100)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(df["ats_score"].mean(), color="red", linestyle="--", label=f"Mean: {df['ats_score'].mean():.1f}")
    axes[0, 0].legend()

    # ── Plot 2: Category Distribution ────────────────────────────────────────
    cat_counts = df["category"].value_counts()
    axes[0, 1].barh(cat_counts.index, cat_counts.values, color="#34C785", edgecolor="white")
    axes[0, 1].set_title("Resume Category Distribution")
    axes[0, 1].set_xlabel("Count")
    for i, v in enumerate(cat_counts.values):
        axes[0, 1].text(v + 1, i, str(v), va="center", fontsize=9)

    # ── Plot 3: Sub-score Correlation ────────────────────────────────────────
    score_cols = ["keyword_score", "formatting_score", "completeness_score",
                  "readability_score", "action_verb_score", "ats_score"]
    corr = df[score_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1, 0], square=True, cbar_kws={"shrink": 0.8})
    axes[1, 0].set_title("Sub-Score Correlation Matrix")

    # ── Plot 4: ATS Score by Category ────────────────────────────────────────
    category_scores = df.groupby("category")["ats_score"].mean().sort_values()
    axes[1, 1].barh(category_scores.index, category_scores.values,
                    color="#F7954F", edgecolor="white")
    axes[1, 1].set_title("Average ATS Score by Category")
    axes[1, 1].set_xlabel("Average ATS Score")
    for i, v in enumerate(category_scores.values):
        axes[1, 1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "training_dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training dashboard saved: {plot_path}")

    # ── Plot 5: Classifier Feature Importance ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Top TF-IDF Features per Category", fontsize=14, fontweight="bold")

    colors = ["#4F8EF7", "#34C785", "#F7954F", "#E74C3C",
              "#9B59B6", "#1ABC9C", "#F39C12", "#2ECC71"]

    for idx, category in enumerate(clf_metrics["classes"][:8]):
        ax = axes[idx // 4][idx % 4]
        features = classifier.get_important_features(category, top_n=8)
        if features:
            names = [f["feature"] for f in features]
            weights = [f["weight"] for f in features]
            ax.barh(names[::-1], weights[::-1], color=colors[idx % len(colors)], alpha=0.85)
        ax.set_title(category, fontsize=10, fontweight="bold")
        ax.set_xlabel("TF-IDF Weight", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    feat_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(feat_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance chart saved: {feat_path}")

    return report


def step6_demo(ats_scorer, classifier, company_recommender, gap_analyzer):
    banner("STEP 6 — End-to-End Demo")

    sample_resume = """
    Rahul Sharma
    rahul.sharma@email.com | +91-9876543210 | linkedin.com/in/rahulsharma

    SUMMARY
    Passionate Data Science enthusiast with 2 years of experience in Python, Machine Learning,
    and SQL. Looking to contribute to data-driven teams.

    SKILLS
    Python, Machine Learning, SQL, Pandas, NumPy, Git, REST API, Flask

    EXPERIENCE
    • Worked on data analysis projects using Python and Pandas.
    • Helped team build machine learning models.
    • Responsible for data cleaning and preprocessing tasks.

    EDUCATION
    B.Tech Computer Science, VTU, 2022 | CGPA: 8.2

    PROJECTS
    • Sales Prediction using Linear Regression.
    """

    sample_jd = """
    Senior Data Scientist — Required Skills: Python, Machine Learning, Deep Learning,
    TensorFlow, PyTorch, NLP, SQL, Feature Engineering, Statistics, AWS, Docker, Spark.
    Preferred: MLflow, Kubernetes, Tableau, Power BI.
    Experience: 2+ years. Must have experience with large datasets and production ML systems.
    """

    print("\n  ── SAMPLE RESUME ─────────────────────────────────────")
    print(sample_resume[:300] + "...")

    # ATS Score
    ats_result = ats_scorer.score(sample_resume, sample_jd)
    print(f"\n  ── ATS SCORING RESULT ────────────────────────────────")
    print(f"  Overall ATS Score: {ats_result.overall_score}/100  [{ats_result.grade()}]")
    print(f"  Sub-Scores:")
    print(f"    Keyword Match:      {ats_result.keyword_score:.1f}/100")
    print(f"    Formatting:         {ats_result.formatting_score:.1f}/100")
    print(f"    Completeness:       {ats_result.completeness_score:.1f}/100")
    print(f"    Readability:        {ats_result.readability_score:.1f}/100")
    print(f"    Action Verbs:       {ats_result.action_verb_score:.1f}/100")
    print(f"\n  Matched Keywords: {ats_result.matched_keywords[:5]}")
    print(f"  Missing Keywords: {ats_result.missing_keywords[:5]}")

    # Improvement tips
    print(f"\n  ── TOP IMPROVEMENT TIPS ──────────────────────────────")
    for i, tip in enumerate(ats_result.improvement_tips[:4], 1):
        print(f"  {i}. [{tip['priority']}] {tip['category']}")
        print(f"     → {tip['recommendation']}")
        print(f"     Expected: {tip['expected_improvement']}")

    # Category prediction
    cat_result = classifier.predict(sample_resume)
    print(f"\n  ── RESUME CATEGORY PREDICTION ────────────────────────")
    print(f"  Predicted: {cat_result['predicted_category']} ({cat_result['confidence']}% confidence)")
    print(f"  Top 3: {[(t['category'], t['probability']) for t in cat_result['top_3']]}")

    # Keyword gap
    gap_result = gap_analyzer.analyze(sample_resume, sample_jd)
    print(f"\n  ── KEYWORD GAP ANALYSIS ─────────────────────────────")
    print(f"  Match %: {gap_result.match_percentage:.1f}%")
    print(f"  Missing ({len(gap_result.missing_keywords)}): "
          f"{[g.keyword for g in gap_result.missing_keywords[:6]]}")

    # Company tips
    print(f"\n  ── COMPANY-SPECIFIC TIPS (Google) ───────────────────")
    tips = company_recommender.get_tips("Google", sample_resume)
    for tip in tips[:3]:
        print(f"  [{tip['priority'].upper()}] {tip['message'][:100]}...")

    print(f"\n  ── COMPANY FIT COMPARISON ───────────────────────────")
    fit = company_recommender.compare_companies(
        sample_resume, ["Google", "Amazon", "TCS", "Infosys", "Flipkart"]
    )
    for co, data in fit.items():
        print(f"  {co:15s}: {data['fit_score']:5.1f}% fit")


def main():
    parser = argparse.ArgumentParser(description="CodeBlaze Resume Screening ML Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to resumes CSV")
    args = parser.parse_args()

    start = time.time()
    banner("🔥 CODEBLAZE — Resume Screening ML Pipeline")
    print("  AI-Powered ATS Score Prediction & Resume Enhancement")
    print(f"  Dataset: Kaggle UpdatedResumeDataSet (synthetic generation)\n")

    # Pipeline
    df = step1_generate_data(args.data)
    df = step2_feature_extraction(df)
    ats_scorer, ats_metrics = step3_train_ats_scorer(df)
    classifier, clf_metrics = step4_train_classifier(df)
    report = step5_evaluate_and_report(df, ats_scorer, classifier, ats_metrics, clf_metrics)

    # Initialize other components for demo
    company_recommender = CompanyRecommender()
    gap_analyzer = KeywordGapAnalyzer()

    step6_demo(ats_scorer, classifier, company_recommender, gap_analyzer)

    elapsed = time.time() - start
    banner(f"✅ Pipeline Complete — {elapsed:.1f}s")
    print(f"  ATS Scorer:       R² = {ats_metrics.get('test_r2', 'N/A')}")
    print(f"  Classifier:       Accuracy = {clf_metrics['test_accuracy']:.4f}")
    print(f"  Reports saved to: reports/")
    print(f"  Models saved to:  models/")
    print()


if __name__ == "__main__":
    main()
