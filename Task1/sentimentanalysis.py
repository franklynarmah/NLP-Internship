import pandas as pd
import nltk
import sklearn
from nltk.stem import   PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"IMDBDataset.csv")

print(df.shape)          
print(df.head())
print(df['sentiment'].value_counts())   # should be ~25000/25000 positive/negative
print(df.isnull().sum())    

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', "", text)
    text = re.sub(r'[^a-z\s]', "", text)
    tokens = word_tokenize(text)
    
    #Remove stopwords from the tokens
    tokens = [word for word in tokens if word not in stop_words]
    
    #Lemmatize each remaining token
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    #Join tokens back into a single cleaned string and return it
    return " ".join(tokens)

# Apply to the whole column 
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Sanity check — compare before/after on one row
print(df['review'].iloc[0][:300])
print(df['cleaned_review'].iloc[0][:300])   

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Compare before/after on a few rows
for i in range(3):
    print("ORIGINAL:", df['review'].iloc[i][:200])
    print("CLEANED:", df['cleaned_review'].iloc[i][:200])
    print("---")         # should be 0 for both columns


print(df['cleaned_review'].isnull().sum())


#TF-IDF (Term Frequency–Inverse Document Frequency)