import os
import sys

def ensure_models():
    model_path = "models/ats_scorer_model.pkl"
    if not os.path.exists(model_path):
        print("Models not found — training now...")
        os.system("python train_pipeline.py")
    else:
        print("Models found — skipping training.")

if __name__ == "__main__":
    ensure_models()