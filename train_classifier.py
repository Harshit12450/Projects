import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\hy903\OneDrive\Documents\documents.txt")

# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = text.lower()                     # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove punctuation/numbers
    text = text.strip()                     # Remove extra spaces
    return text

data["text"] = data["text"].apply(clean_text)

# -----------------------------
# Display Dataset Information
# -----------------------------
print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nCategory Distribution:")
print(data["category"].value_counts())

# -----------------------------
# Features and Labels
# -----------------------------
X = data["text"]
y = data["category"]

# -----------------------------
# Convert Text into Numbers
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

# -----------------------------
# Make Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluate Model
# -----------------------------
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "document_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and Vectorizer saved successfully!")