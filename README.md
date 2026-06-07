# 📰 Fake News Prediction Using Machine Learning

Fake News Prediction is a Machine Learning project that uses NLP techniques and Logistic Regression to classify news articles as Real or Fake. The model is trained on labeled news data and can predict the authenticity of new articles through a Python-based prediction script.

## 📖 Overview

With the rapid spread of misinformation online, identifying fake news has become increasingly important. This project leverages NLP and Machine Learning to analyze news content and predict whether an article is genuine or fake.

## ✨ Features

* News authenticity prediction (Real/Fake)
* Text preprocessing using NLTK
* TF-IDF feature extraction
* Logistic Regression classifier
* Model evaluation with accuracy metrics
* Easy-to-use prediction script
* Modular project structure

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib
* TF-IDF Vectorizer
* Logistic Regression

## 📂 Project Structure

```text
fake-news-prediction/
│
├── data/
│   └── fake_news_sample_dataset.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Feature Extraction using TF-IDF
5. Model Training
6. Model Evaluation
7. Fake News Prediction

## 🤖 Model Details

### Algorithm

* Logistic Regression

### Feature Extraction

* TF-IDF Vectorization

### Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

## 🚀 Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Dharshan891/fake-news-prediction.git
cd fake-news-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

```bash
python src/train.py
```

## 🔍 Predict News

```bash
python src/predict.py
```

## 📈 Sample Output

```text
Enter News Article:
Breaking news article text...

Prediction: Real News ✅
```

or

```text
Prediction: Fake News ❌
```

## 👨‍💻 Team Members

* T. C. Dharshan (2023006380)
* D. Mukesh (2023003986)
* Sameer (2023004759)

## 🔮 Future Enhancements

* Deep Learning Models (LSTM, BERT)
* Real-Time News Verification
* Web Application Deployment
* REST API Integration
* News Source Credibility Analysis

## 📜 License

This project is developed for educational and research purposes.

## ⭐ Support

If you found this project helpful, please consider giving it a star on GitHub.
