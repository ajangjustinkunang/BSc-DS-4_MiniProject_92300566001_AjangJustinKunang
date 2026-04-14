"""
Spam Detection Machine Learning Model
======================================
This module implements a complete ML pipeline for spam detection on the SMS
spam dataset. It includes text preprocessing, feature extraction, feature
selection, model training, and evaluation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class SpamDetectionModel:
    """
    Machine Learning model for spam detection using text data.
    """

    def __init__(self, model_path="trained_model.pkl", vectorizer_path="vectorizer.pkl", selector_path="feature_selector.pkl"):
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.selector_path = selector_path

    def load_data(self, data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, encoding="latin-1")
        df = df.rename(columns={"v1": "label", "v2": "message"})
        df = df[["label", "message"]].dropna(subset=["message"])
        df["label"] = df["label"].map({"ham": 0, "spam": 1})

        if df["label"].isnull().any():
            raise ValueError("Unexpected label values found in dataset.")

        X = df["message"].values
        y = df["label"].values

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}\n")
        return X, y

    def preprocess_data(self, X):
        print("\n--- PREPROCESSING STAGE ---")
        print("Converting raw text into TF-IDF feature vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X_tfidf = self.vectorizer.fit_transform(X)
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")
        return X_tfidf

    def select_features(self, X, y, n_features=2000):
        print("\n--- FEATURE SELECTION STAGE ---")
        print(f"Selecting top {n_features} text features using chi2...")
        self.feature_selector = SelectKBest(chi2, k=min(n_features, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        print(f"Selected feature matrix shape: {X_selected.shape}")
        return X_selected

    def train_model(self, X_train, y_train):
        print("\n--- MODEL TRAINING STAGE ---")
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate_model(self, X_test, y_test):
        print("\n--- MODEL EVALUATION STAGE ---")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Test Set Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

        print("\nCross-validation scores (5-fold):")
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring="accuracy")
        print(f"  Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
        }

    def save_model(self):
        print("\n--- SAVING MODEL COMPONENTS ---")
        for path in [self.model_path, self.vectorizer_path, self.selector_path]:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.selector_path, "wb") as f:
            pickle.dump(self.feature_selector, f)
        print(f"Saved model to {self.model_path}")
        print(f"Saved vectorizer to {self.vectorizer_path}")
        print(f"Saved selector to {self.selector_path}")

    def load_model(self):
        print("Loading model components from disk...")
        if not all(os.path.exists(p) for p in [self.model_path, self.vectorizer_path, self.selector_path]):
            print("Model components not found.")
            return False
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.selector_path, "rb") as f:
            self.feature_selector = pickle.load(f)
        print("Model components loaded successfully.")
        return True

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        X_selected = self.feature_selector.transform(X)
        prediction = self.model.predict(X_selected)
        return prediction[0]

    def predict_proba(self, texts):
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        X_selected = self.feature_selector.transform(X)
        return self.model.predict_proba(X_selected)[0]


def main():
    print("=" * 60)
    print("SPAM DETECTION MODEL - COMPLETE ML PIPELINE")
    print("=" * 60)

    data_path = "data/spam.csv"
    model_path = "trained_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    selector_path = "feature_selector.pkl"

    model = SpamDetectionModel(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        selector_path=selector_path,
    )

    X, y = model.load_data(data_path)

    print("\n--- TRAIN-TEST SPLIT ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

    X_train_tfidf = model.preprocess_data(X_train)
    X_test_tfidf = model.vectorizer.transform(X_test)

    X_train_selected = model.select_features(X_train_tfidf, y_train, n_features=2000)
    X_test_selected = model.feature_selector.transform(X_test_tfidf)

    model.train_model(X_train_selected, y_train)

    metrics = model.evaluate_model(X_test_selected, y_test)

    model.save_model()

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
