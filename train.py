import os
import re
import string
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [
        lemm.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return ' '.join(tokens)

def train_model():
    csv_file = "spam.csv" if os.path.exists("spam.csv") else "SMS.csv"
    try:
        df = pd.read_csv(csv_file, encoding="latin-1")
    except Exception:
        df = pd.read_csv(csv_file, encoding="utf-8")

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']

    df = df.drop_duplicates().reset_index(drop=True)
    df['clean_message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 1. SPLIT FIRST (LEAK-FREE)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.20, random_state=42, stratify=df['label']
    )
    
    # 2. OVERSAMPLE TRAINING SET ONLY
    train_df = pd.DataFrame({'clean_message': X_train, 'label': y_train})
    train_ham = train_df[train_df.label == 0]
    train_spam = train_df[train_df.label == 1]
    
    train_spam_oversampled = resample(train_spam, replace=True, n_samples=len(train_ham), random_state=42)
    train_balanced = pd.concat([train_ham, train_spam_oversampled]).reset_index(drop=True)
    
    X_train_bal = train_balanced['clean_message']
    y_train_bal = train_balanced['label']
    
    print("=== SMS Spam SVM Classifier Training (Leak-Free Pipeline) ===")
    print("Raw Train counts:", y_train.value_counts().to_dict())
    print("Balanced Train counts:", y_train_bal.value_counts().to_dict())
    print("Pure Test counts:", y_test.value_counts().to_dict())
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svm', LinearSVC(random_state=42))
    ])
    
    param_grid = {'svm__C': [0.1, 1, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train_bal, y_train_bal)
    
    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    
    print("\nBest Parameters:", grid.best_params_)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1-Score :", f1_score(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    
    scores = cross_val_score(best_model, X_train_bal, y_train_bal, cv=5)
    print("5-Fold CV Scores (on training folds):", scores)
    print("Mean CV Accuracy (on training folds):", scores.mean())
    
    joblib.dump(best_model, "sms_spam_pipeline.pkl")
    print("\nModel saved to sms_spam_pipeline.pkl!")

if __name__ == "__main__":
    train_model()
