import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from preprocess import load_and_validate_data
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    extract_top_tfidf_words,
    plot_top_keywords_summary
)

CATEGORY_DESCRIPTIONS = {
    "Billing & Payment": "Issues involving charges, invoices, payments and billing.",
    "Delivery": "Issues involving shipment, courier, tracking and delivery.",
    "Product Issue": "Issues involving damaged, defective, incorrect or poor-quality products.",
    "Account & Login": "Issues involving passwords, login, verification and account access.",
    "Technical Issue": "Issues involving application, website, errors, crashes and technical failures.",
    "Refund": "Issues involving pending, incorrect or missing refunds.",
    "Cancellation": "Requests or problems involving cancellation of orders, subscriptions or bookings.",
    "General Inquiry": "General information requests that do not represent a specific complaint category."
}


def run_pipeline(data_path: str = "data/complaint_classification_dataset_1600.csv"):
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / data_path
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PS-01: SMART COMPLAINT CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)

    print(f"\n[1/7] Loading dataset from: {data_file}")
    df, stats = load_and_validate_data(str(data_file))
    print(f" -> Successfully loaded {stats['clean_rows']} records across {len(stats['category_counts'])} categories.")
    print(f" -> Missing values found: {sum(stats['missing_counts'].values())}")
    print(f" -> Duplicate complaints removed: {stats['duplicate_complaints']}")
    print(f" -> Balanced dataset confirmed: {stats['is_balanced']}")

    print("\n[2/7] Generating Exploratory Data Analysis plots...")
    plt.figure(figsize=(10, 5))
    order = sorted(df["category"].unique())
    ax = sns.countplot(data=df, x="category", order=order, hue="category", legend=False, palette="Blues_r")
    plt.title("Distribution of Complaints Across Categories (N=1600)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Complaint Category", fontsize=10)
    plt.ylabel("Number of Complaints", fontsize=10)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    dist_plot_path = outputs_dir / "category_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f" -> Category distribution chart saved to {dist_plot_path.name}")

    print("\n[3/7] Performing 80/20 Stratified Train/Test split...")
    X = df["cleaned_complaint"]
    y = df["category"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f" -> Training set size: {len(X_train_raw)} samples (80%)")
    print(f" -> Testing set size:  {len(X_test_raw)} samples (20%)")

    print("\n[4/7] Fitting TF-IDF Vectorizer ONLY on training set (Preventing Data Leakage)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f" -> TF-IDF feature vocabulary size: {X_train_tfidf.shape[1]} features extracted.")

    top_words_by_cat = extract_top_tfidf_words(vectorizer, X_train_tfidf, y_train, top_n=10)
    keywords_plot_path = outputs_dir / "common_words.png"
    plot_top_keywords_summary(top_words_by_cat, str(keywords_plot_path))
    print(f" -> Category TF-IDF distinctive terms extracted and saved to {keywords_plot_path.name}")

    print("\n[5/7] Training Model 1: Logistic Regression (max_iter=2000, random_state=42)...")
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    lr_metrics = compute_metrics(y_test, y_pred_lr, "Logistic Regression")

    print("[6/7] Training Model 2: Linear Support Vector Machine (random_state=42)...")
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    svm_metrics = compute_metrics(y_test, y_pred_svm, "Linear SVM")

    categories_list = sorted(list(CATEGORY_DESCRIPTIONS.keys()))
    cm_lr_path = outputs_dir / "confusion_matrix_lr.png"
    plot_confusion_matrix(y_test, y_pred_lr, categories_list, "Confusion Matrix - Logistic Regression", str(cm_lr_path))

    cm_svm_path = outputs_dir / "confusion_matrix_svm.png"
    plot_confusion_matrix(y_test, y_pred_svm, categories_list, "Confusion Matrix - Linear SVM", str(cm_svm_path))

    comparison_chart_path = outputs_dir / "model_comparison.png"
    plot_model_comparison([lr_metrics, svm_metrics], str(comparison_chart_path))

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (TEST SET, 320 SAMPLES)")
    print("=" * 60)

    comparison_df = pd.DataFrame([lr_metrics, svm_metrics])[
        ["model_name", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"]
    ]
    print(comparison_df.to_string(index=False))

    print("\nClassification Report: Logistic Regression")
    print(classification_report(y_test, y_pred_lr, digits=4))

    print("\nClassification Report: Linear SVM")
    print(classification_report(y_test, y_pred_svm, digits=4))

    print("[7/7] Dynamically selecting final model based on Test Weighted F1-score...")
    if svm_metrics["f1_weighted"] > lr_metrics["f1_weighted"]:
        best_model_name = "Linear SVM"
        best_model = svm_model
        best_metrics = svm_metrics
    else:
        best_model_name = "Logistic Regression"
        best_model = lr_model
        best_metrics = lr_metrics

    print(f" -> Selected Model: {best_model_name} (F1 Score: {best_metrics['f1_weighted']:.4f})")

    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")
    joblib.dump(lr_model, models_dir / "logistic_regression.pkl")
    joblib.dump(svm_model, models_dir / "linear_svm.pkl")
    joblib.dump(best_model, models_dir / "final_model.pkl")

    lr_report_dict = classification_report(y_test, y_pred_lr, output_dict=True)
    svm_report_dict = classification_report(y_test, y_pred_svm, output_dict=True)

    metadata = {
        "best_model_name": best_model_name,
        "selected_metric": "f1_weighted",
        "best_metrics": best_metrics,
        "all_metrics": [lr_metrics, svm_metrics],
        "per_class_reports": {
            "Logistic Regression": lr_report_dict,
            "Linear SVM": svm_report_dict
        },
        "categories": categories_list,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "top_keywords_by_category": top_words_by_cat,
        "dataset_stats": {
            "total_records": stats["clean_rows"],
            "num_categories": len(categories_list),
            "is_balanced": stats["is_balanced"],
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "category_counts": stats["category_counts"]
        },
        "tfidf_config": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "sublinear_tf": True
        }
    }

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nArtifacts successfully saved:")
    print(f" - Vectorizer:  {models_dir / 'tfidf_vectorizer.pkl'}")
    print(f" - LR Model:    {models_dir / 'logistic_regression.pkl'}")
    print(f" - SVM Model:   {models_dir / 'linear_svm.pkl'}")
    print(f" - Final Model: {models_dir / 'final_model.pkl'}")
    print(f" - Metadata:    {models_dir / 'model_metadata.json'}")
    print(f" - Plots:       {outputs_dir}")
    print("\nTraining and evaluation pipeline finished successfully!")


if __name__ == "__main__":
    run_pipeline()
