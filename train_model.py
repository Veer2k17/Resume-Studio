
from pathlib import Path
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils.preprocessing import clean_text
from utils.model import train_classifier, save_model


DATA_PATH = Path(__file__).resolve().parent / "data" / "resumes_cleaned.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"Resume", "Category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain Resume and Category columns."
        )

    df = df.dropna(subset=["Resume", "Category"]).copy()
    df["Resume"] = df["Resume"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[(df["Resume"] != "") & (df["Category"] != "")]

    df["CleanResume"] = df["Resume"].apply(clean_text)
    df = df[df["CleanResume"] != ""]

    # Avoid duplicate resume text appearing in both train and test.
    df = df.drop_duplicates(subset=["CleanResume"])

    counts = df["Category"].value_counts()

    if len(counts) < 2:
        raise ValueError("At least two job categories are required.")

    if (counts < 2).any():
        raise ValueError(
            "Each category needs at least two distinct resumes "
            "for a stratified train/test split."
        )

    X = df["Resume"]
    y = df["Category"]

    test_count = max(math.ceil(len(df) * 0.20), y.nunique())

    if len(df) - test_count < y.nunique():
        raise ValueError(
            "Dataset is too small for a stratified split. "
            "Add more distinct resumes to each category."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=42,
        stratify=y
    )

    # Evaluate on data not used to fit this model.
    evaluation_model, evaluation_vectorizer = train_classifier(
        X_train.tolist(),
        y_train.tolist()
    )

    X_test_features = evaluation_vectorizer.transform(
        X_test.apply(clean_text)
    )
    predictions = evaluation_model.predict(X_test_features)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Total distinct resumes: {len(df)}")
    print(f"Training resumes: {len(X_train)}")
    print(f"Testing resumes: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=evaluation_model.classes_
    ))
    print("Class order:", list(evaluation_model.classes_))

    # Fit the deployable model on all available labeled data.
    final_model, final_vectorizer = train_classifier(
        X.tolist(),
        y.tolist()
    )
    save_model(final_model, final_vectorizer)

    print("\nFinal model and vectorizer saved in models/.")


if __name__ == "__main__":
    main()
