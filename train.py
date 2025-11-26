import os, re, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/cleaned_ai_human.csv"
MODEL_PATH = "models/ai_detector_model.pkl"
VECT_PATH = "models/tfidf_vectorizer.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean labels
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"].isin(["ai", "human"])]

# Clean function
def clean_text(s):
    s = str(s)
    s = re.sub(r"\[.*?\]", " ", s)   # remove templates
    s = re.sub(r"\b(ai|human|text)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d{1,3}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text_clean"] = df["text"].apply(clean_text)

# Prepare data
X = df["text_clean"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1,2)
)

X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train_tf, y_train)

# Evaluate
y_pred = model.predict(X_test_tf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECT_PATH)

print("\nSaved model →", MODEL_PATH)
print("Saved vectorizer →", VECT_PATH)
