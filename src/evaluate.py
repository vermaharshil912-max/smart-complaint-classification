import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Any


def compute_metrics(y_true, y_pred, model_name: str) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_macro": round(float(f1_m), 4),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
    title: str,
    save_path: str
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11, labelpad=10)
    plt.ylabel("True Category", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(metrics_list: List[Dict[str, Any]], save_path: str):
    df = pd.DataFrame(metrics_list)
    metrics_to_plot = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(
        x - width / 2,
        [df.loc[0, m] for m in metrics_to_plot],
        width,
        label=df.loc[0, "model_name"],
        color="#2b5c8f"
    )
    rects2 = ax.bar(
        x + width / 2,
        [df.loc[1, m] for m in metrics_to_plot],
        width,
        label=df.loc[1, "model_name"],
        color="#48a9a6"
    )

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_top_tfidf_words(
    vectorizer,
    X_tfidf,
    y: pd.Series,
    top_n: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    categories = sorted(y.unique())
    top_words_by_category = {}

    for cat in categories:
        cat_indices = (y == cat).values
        cat_matrix = X_tfidf[cat_indices]
        mean_scores = np.asarray(cat_matrix.mean(axis=0)).ravel()
        top_indices = mean_scores.argsort()[::-1][:top_n]

        top_words = [
            {"word": feature_names[idx], "tfidf_score": round(float(mean_scores[idx]), 4)}
            for idx in top_indices
        ]
        top_words_by_category[cat] = top_words

    return top_words_by_category


def plot_top_keywords_summary(top_words_by_cat: Dict[str, List[Dict[str, Any]]], save_path: str):
    categories = list(top_words_by_cat.keys())
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        items = top_words_by_cat[cat][:8] 
        words = [item["word"] for item in items][::-1]
        scores = [item["tfidf_score"] for item in items][::-1]

        ax.barh(words, scores, color="#347474")
        ax.set_title(cat, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Mean TF-IDF Weight", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Distinctive TF-IDF Terms by Complaint Category", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()