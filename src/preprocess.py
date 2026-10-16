import re
import string
import pandas as pd
from typing import Tuple, Dict, Any


def clean_text(text: Any) -> str:

    if text is None or pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_validate_data(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(filepath)

    required_cols = {"id", "complaint", "category"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    exact_duplicates = df.duplicated().sum()
    duplicate_complaints = df.duplicated(subset=["complaint"]).sum()
    category_counts = df["category"].value_counts().to_dict()

    if exact_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    if duplicate_complaints > 0:
        df = df.drop_duplicates(subset=["complaint"]).reset_index(drop=True)

    df["cleaned_complaint"] = df["complaint"].apply(clean_text)

    df = df[df["cleaned_complaint"].str.strip() != ""].reset_index(drop=True)

    stats = {
        "initial_rows": total_rows,
        "clean_rows": len(df),
        "missing_counts": missing_counts,
        "exact_duplicates": int(exact_duplicates),
        "duplicate_complaints": int(duplicate_complaints),
        "category_counts": category_counts,
        "is_balanced": len(set(category_counts.values())) <= 1
    }

    return df, stats


if __name__ == "__main__":
    import os
    dataset_path = os.path.join("data", "complaint_classification_dataset_1600.csv")
    if os.path.exists(dataset_path):
        df, stats = load_and_validate_data(dataset_path)
        print("=== Dataset Validation Stats ===")
        print(f"Total Rows: {stats['clean_rows']}")
        print(f"Missing Values: {stats['missing_counts']}")
        print(f"Duplicate Complaints: {stats['duplicate_complaints']}")
        print(f"Is Balanced: {stats['is_balanced']}")
        print("Category Distribution:")
        for cat, cnt in stats["category_counts"].items():
            print(f"  {cat}: {cnt}")
        print("\nSample Cleaned Text:")
        print("Original:", df["complaint"].iloc[0])
        print("Cleaned: ", df["cleaned_complaint"].iloc[0])
    else:
        print(f"Dataset not found at: {dataset_path}")
