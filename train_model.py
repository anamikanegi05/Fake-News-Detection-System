import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

FAKE_PATH = "dataset/Fake.csv"
REAL_PATH = "dataset/True.csv"

if not os.path.exists(FAKE_PATH) or not os.path.exists(REAL_PATH):
    raise FileNotFoundError(
        "Please place Fake.csv and True.csv inside the dataset folder."
    )


def clean_text(text):
    """Clean news text before processing."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


fake_news = pd.read_csv(FAKE_PATH)
real_news = pd.read_csv(REAL_PATH)

fake_news["label"] = 0
real_news["label"] = 1

data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)
data = data[["text", "label"]].dropna()
data = data.drop_duplicates(subset=["text"])

data["text"] = data["text"].apply(clean_text)

data = data[data["text"].str.len() > 0]

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("Training the model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")

print("\nModel saved successfully!")
