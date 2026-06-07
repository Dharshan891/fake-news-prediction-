import joblib
import os
from preprocessing import clean_text

models_dir = r"C:\Users\manig\OneDrive\Desktop\fake news predection\models"

model = joblib.load(os.path.join(models_dir, "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))

while True:
    news = input("\nEnter news text (type 'quit' to exit): ")

    if news.lower() == "quit":
        print("Exiting...")
        break

    cleaned_news = clean_text(news)

    news_vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vector)

    print("Prediction:", prediction[0])