import pandas as pd
import nltk
import sklearn
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import   PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("news_articles.csv")
df = df.drop(columns=['url'])
print(df.isnull().sum())    
print(df.shape)
print(df.head())
print(df['category'].value_counts())


stop_words = set(stopwords.words("English"))
Lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', "", text)
    text = re.sub(r'[^a-z\s]', "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [Lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['cleaned_review'] = df['body'].apply(preprocess_text)
print(df.head())

#Vectorization
Vectorizer = TfidfVectorizer(max_features=5000)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'],
    df['category'],
    test_size=0.2,
    random_state=42
)

X_train_vec = Vectorizer.fit_transform(X_train)
X_test_vec = Vectorizer.transform(X_test)

print(df['category'].value_counts())
print("-----------------------------")
print("X_train_vec shape:", X_train_vec.shape)
print("X_test_vec shape:", X_test_vec.shape)
print("-----------------------------")
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))
print("-----------------------------")
print("Sample vocabulary words:", list(Vectorizer.vocabulary_.keys())[:20])
print("Vocabulary size:", len(Vectorizer.vocabulary_))
print("-----------------------------")



