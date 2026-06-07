import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
# Dataset should contain columns: 'text' and 'label'
fake = pd.read_csv(r"C:\Users\manig\Downloads\news\Fake.csv")
real = pd.read_csv(r"C:\Users\manig\Downloads\news\True.csv")

fake["label"] = "FAKE"
real["label"] = "REAL"

df = pd.concat([fake, real], ignore_index=True)

print(df.head())

# Features and Labels
X = df['text']
y = df['label']

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict custom news
news = ["Breaking: Scientists discover a new planet similar to Earth."]
news_tfidf = vectorizer.transform(news)

prediction = model.predict(news_tfidf)

if prediction[0] == 1:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")