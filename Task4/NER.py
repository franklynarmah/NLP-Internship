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
import seaborn as sns
import numpy as np
import spacy
from spacy import displacy

df = pd.read_csv("news_articles.csv")
print(df.head())
df = df.drop(columns=['url', 'category'])
print(df.head())

nlp = spacy.load("en_core_web_sm")

#NER Model Test
for i in range(5):
    text = df['body'].iloc[i]
    doc = nlp(text)
    
    print(f"--- Article {i} ---------------------------------------")
    
    for ent in doc.ents:
        print(ent.text, "->", ent.label_)

text = df['body'].iloc[0]
doc = nlp(text)

displacy.serve(doc, style="ent", port= 5001)

