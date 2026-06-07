import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

from preprocessing import clean_text

# Load dataset
fake_df = pd.read_csv(r"C:\Users\manig\Downloads\news\Fake.csv")
real_df = pd.read_csv(r"C:\Users\manig\Downloads\news\True.csv")

fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

df = pd.concat([fake_df, real_df], ignore_index=True)

df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)

model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nSample Test Prediction:")

sample_text = X_test.iloc[0]
sample_actual = y_test.iloc[0]

sample_vector = vectorizer.transform([sample_text])
sample_predicted = model.predict(sample_vector)[0]

print("Actual Label:", sample_actual)
print("Predicted Label:", sample_predicted)
print("Sample Text:", sample_text[:200])

# Save model
import os

models_dir = r"C:\Users\manig\OneDrive\Desktop\fake news predection\models"

os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))

print("Model Saved Successfully")

print(df["label"].value_counts())