import re
import joblib

# Load saved model and vectorizer
model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


while True:
    text = input("\nEnter a document (or type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    text = clean_text(text)

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    print("Predicted Category:", prediction[0])