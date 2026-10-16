import sys
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class ComplaintClassifier:
    def __init__(self, models_path: Path = MODELS_DIR):
        self.models_path = models_path
        self.vectorizer = None
        self.model = None
        self.metadata = None
        self.lr_model = None
        self.svm_model = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        vec_file = self.models_path / "tfidf_vectorizer.pkl"
        model_file = self.models_path / "final_model.pkl"
        lr_file = self.models_path / "logistic_regression.pkl"
        svm_file = self.models_path / "linear_svm.pkl"
        meta_file = self.models_path / "model_metadata.json"

        if not (vec_file.exists() and model_file.exists()):
            self.is_loaded = False
            return

        self.vectorizer = joblib.load(vec_file)
        self.model = joblib.load(model_file)
        self.lr_model = joblib.load(lr_file) if lr_file.exists() else None
        self.svm_model = joblib.load(svm_file) if svm_file.exists() else None

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.is_loaded = True

    def predict(self, complaint_text: str) -> Dict[str, Any]:
        if not complaint_text or str(complaint_text).strip() == "":
            return {
                "status": "error",
                "message": "Please enter a complaint before classification.",
                "predicted_category": None
            }

        if not self.is_loaded:
            self._load_artifacts()
            if not self.is_loaded:
                return {
                    "status": "error",
                    "message": "Trained model artifacts not found. Please run 'python src/train.py' first.",
                    "predicted_category": None
                }

        cleaned = clean_text(complaint_text)
        if not cleaned:
            return {
                "status": "error",
                "message": "Complaint text contained only symbols or punctuation. Please enter meaningful text.",
                "predicted_category": None
            }

        features = self.vectorizer.transform([cleaned])

        predicted_category = self.model.predict(features)[0]

        categories = list(self.model.classes_)
        all_scores = {}
        score_type = "Confidence Score"
        confidence_val = 0.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            pred_idx = categories.index(predicted_category)
            confidence_val = float(probs[pred_idx])
            score_type = "Predicted Probability"
            for cat, prob in zip(categories, probs):
                all_scores[cat] = round(float(prob), 4)
        elif hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(features)[0]
            exp_d = np.exp(decisions - np.max(decisions))
            norm_scores = exp_d / exp_d.sum()

            pred_idx = categories.index(predicted_category)
            confidence_val = float(norm_scores[pred_idx])
            score_type = "Decision Score (Normalized)"

            for cat, score in zip(categories, norm_scores):
                all_scores[cat] = round(float(score), 4)

        both_models_data = {}
        if self.lr_model is not None and self.svm_model is not None:
            lr_pred = self.lr_model.predict(features)[0]
            lr_probs = self.lr_model.predict_proba(features)[0]
            lr_classes = list(self.lr_model.classes_)
            lr_conf = float(lr_probs[lr_classes.index(lr_pred)])
            lr_scores = {c: round(float(p), 4) for c, p in zip(lr_classes, lr_probs)}

            svm_pred = self.svm_model.predict(features)[0]
            svm_dec = self.svm_model.decision_function(features)[0]
            svm_classes = list(self.svm_model.classes_)
            exp_s = np.exp(svm_dec - np.max(svm_dec))
            svm_norm = exp_s / exp_s.sum()
            svm_conf = float(svm_norm[svm_classes.index(svm_pred)])
            svm_scores = {c: round(float(s), 4) for c, s in zip(svm_classes, svm_norm)}

            both_models_data = {
                "logistic_regression": {
                    "model_name": "Logistic Regression",
                    "predicted_category": lr_pred,
                    "score_type": "Predicted Probability",
                    "score_value": round(lr_conf, 4),
                    "score_percentage": f"{lr_conf * 100:.1f}%",
                    "all_scores": lr_scores
                },
                "linear_svm": {
                    "model_name": "Linear SVM",
                    "predicted_category": svm_pred,
                    "score_type": "Normalized Decision Score",
                    "score_value": round(svm_conf, 4),
                    "score_percentage": f"{svm_conf * 100:.1f}%",
                    "all_scores": svm_scores
                },
                "models_agree": bool(lr_pred == svm_pred),
                "consensus_message": f"Both models independently agree on '{lr_pred}'" if lr_pred == svm_pred else f"Models differ: LR predicts '{lr_pred}', Linear SVM predicts '{svm_pred}'"
            }

        descriptions = self.metadata.get("category_descriptions", {})
        description = descriptions.get(predicted_category, "No description available.")

        feature_names = set(self.vectorizer.get_feature_names_out())
        words_in_text = cleaned.split()
        matched_tokens = [w for w in words_in_text if w in feature_names]

        return {
            "status": "success",
            "original_complaint": complaint_text,
            "cleaned_complaint": cleaned,
            "predicted_category": predicted_category,
            "category_description": description,
            "score_type": score_type,
            "score_value": round(confidence_val, 4),
            "score_percentage": f"{confidence_val * 100:.1f}%",
            "all_scores": all_scores,
            "matched_keywords": matched_tokens[:8],
            "model_name": self.metadata.get("best_model_name", type(self.model).__name__),
            "both_models": both_models_data
        }


_classifier_instance = None


def predict_complaint(complaint_text: str) -> Dict[str, Any]:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ComplaintClassifier()
    return _classifier_instance.predict(complaint_text)


if __name__ == "__main__":
    test_queries = [
        "My payment was deducted twice for the same order.",
        "My package shows delivered but I never received it.",
        "The product arrived broken and has missing parts.",
        "I forgot my password and cannot log into my account.",
        "The application crashes whenever I try to open it.",
        "I requested a refund several days ago but haven't received it.",
        "I want to cancel my subscription.",
        "What plans and services do you offer?"
    ]

    print("=== Testing Standalone Predict Pipeline ===")
    for query in test_queries:
        res = predict_complaint(query)
        if res["status"] == "success":
            print(f"\nComplaint: '{query}'")
            print(f" -> Predicted Category: {res['predicted_category']}")
            print(f" -> Score ({res['score_type']}): {res['score_percentage']}")
        else:
            print(f" -> Error: {res['message']}")
