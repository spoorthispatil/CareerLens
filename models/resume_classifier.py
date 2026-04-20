"""
models/resume_classifier.py
────────────────────────────────────────────────────────────────────────────────
Resume Category Classifier
────────────────────────────────────────────────────────────────────────────────
Classifies a resume into one of 8 job categories using TF-IDF + LogisticRegression.
Also supports a BERT-based variant when transformers are available.

Categories: Data Science, Web Development, DevOps, Android Developer,
            HR, Java Developer, Testing, Business Analyst
────────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = os.path.join(os.path.dirname(__file__), "resume_classifier.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")


class ResumeClassifier:
    """
    Multi-class resume category classifier.
    Pipeline: TF-IDF → Logistic Regression (best for text classification).
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=8000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def train(self, df: pd.DataFrame, text_col: str = "resume_text",
              label_col: str = "category") -> dict:
        """
        Train the classifier.
        Returns evaluation metrics.
        """
        print("\n  Training Resume Category Classifier...")

        # Ensure numpy arrays (fixes PyArrow/pandas dtype issues with real Kaggle data)
        X = df[text_col].astype(str).to_numpy()
        y_raw = df[label_col].astype(str).to_numpy()

        # Drop categories with fewer than 2 samples (can't stratify-split them)
        from collections import Counter
        counts = Counter(y_raw)
        rare = {cat for cat, cnt in counts.items() if cnt < 2}
        if rare:
            print(f"    ⚠  Dropping {len(rare)} rare categories (< 2 samples): {rare}")
            mask = np.array([c not in rare for c in y_raw])
            X, y_raw = X[mask], y_raw[mask]

        y = self.label_encoder.fit_transform(y_raw)

        # Use stratify only if every class has ≥ 2 samples
        min_class_count = min(Counter(y_raw).values())
        use_stratify = y if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=use_stratify
        )

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"    CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit
        self.pipeline.fit(X_train, y_train)
        self._fitted = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        test_acc = (y_pred == y_test).mean()
        print(f"    Test Accuracy: {test_acc:.4f}")

        classes = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=classes,
            output_dict=True
        )

        # Save
        joblib.dump(self.pipeline, MODEL_PATH)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        print(f"    ✓ Model saved to {MODEL_PATH}")

        return {
            "cv_accuracy": float(cv_scores.mean()),
            "test_accuracy": float(test_acc),
            "classification_report": report,
            "classes": list(classes),
        }

    def predict(self, resume_text: str) -> dict:
        """Predict category with confidence scores."""
        if not self._fitted:
            self.load_model()

        proba = self.pipeline.predict_proba([resume_text])[0]
        classes = self.label_encoder.classes_
        top_indices = proba.argsort()[::-1][:3]

        return {
            "predicted_category": classes[top_indices[0]],
            "confidence": round(float(proba[top_indices[0]]) * 100, 1),
            "top_3": [
                {
                    "category": classes[i],
                    "probability": round(float(proba[i]) * 100, 1)
                }
                for i in top_indices
            ]
        }

    def load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self._fitted = True
            print(f"  ✓ Classifier loaded from {MODEL_PATH}")
        else:
            raise FileNotFoundError("Model not found. Run train() first.")

    def plot_confusion_matrix(self, df: pd.DataFrame, save_path: str = None):
        """Plot confusion matrix for evaluation."""
        X = df["resume_text"].values
        y_true = self.label_encoder.transform(df["category"].values)
        y_pred = self.pipeline.predict(X)
        classes = self.label_encoder.classes_

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes
        )
        plt.title("Resume Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Category")
        plt.xlabel("Predicted Category")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def get_important_features(self, category: str, top_n: int = 15) -> list:
        """Get most important TF-IDF features for a category."""
        if not self._fitted:
            self.load_model()

        clf = self.pipeline.named_steps["clf"]
        tfidf = self.pipeline.named_steps["tfidf"]
        feature_names = tfidf.get_feature_names_out()

        try:
            class_idx = list(self.label_encoder.classes_).index(category)
            coefs = clf.coef_[class_idx]
            top_indices = coefs.argsort()[-top_n:][::-1]
            return [
                {"feature": feature_names[i], "weight": round(float(coefs[i]), 4)}
                for i in top_indices
            ]
        except (ValueError, IndexError):
            return []


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generate_dataset import generate_resumes

    print("Generating training data...")
    df = generate_resumes(500)

    classifier = ResumeClassifier()
    metrics = classifier.train(df)

    print(f"\nClassification Results:")
    print(f"  CV Accuracy:   {metrics['cv_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")

    # Test prediction
    test_resume = """
    Python Machine Learning TensorFlow PyTorch Deep Learning NLP Computer Vision
    Data Science SQL Pandas NumPy Scikit-learn Feature Engineering Statistics
    Developed ML models achieving 92% accuracy. Deployed on AWS SageMaker.
    """
    result = classifier.predict(test_resume)
    print(f"\nTest Prediction:")
    print(f"  Category: {result['predicted_category']} ({result['confidence']}% confidence)")
    print(f"  Top 3: {result['top_3']}")

    # Feature importance
    print(f"\nTop features for 'Data Science':")
    for feat in classifier.get_important_features("Data Science", top_n=8):
        print(f"  {feat['feature']}: {feat['weight']}")