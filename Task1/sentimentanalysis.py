import pandas as pd
import nltk
import sklearn
import re
from sklearn.linear_model import LogisticRegression
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

# Sanity check
# Compare before/after on a few rows
for i in range(3):
    print("ORIGINAL:", df['review'].iloc[i][:200])
    print("CLEANED:", df['cleaned_review'].iloc[i][:200])
    print("---")         
print(df['cleaned_review'].isnull().sum())


#TF-IDF
Vectorizer = TfidfVectorizer(max_features=5000)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42
)

X_train_vec = Vectorizer.fit_transform(X_train)
X_test_vec = Vectorizer.transform(X_test)


# Sanity checks after vectorization

#Check the shape of vectorized matrices
print("X_train_vec shape:", X_train_vec.shape)
print("X_test_vec shape:", X_test_vec.shape)

# 2. Check the shape/length of your labels matches
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))

# 3. Peek at the actual vocabulary the vectorizer learned
print("Sample vocabulary words:", list(Vectorizer.vocabulary_.keys())[:20])

# 4. Confirm vocabulary size matches your max_features setting
print("Vocabulary size:", len(Vectorizer.vocabulary_))


#the model
from sklearn.linear_model import LogisticRegression

# 1. Create the model
model = LogisticRegression(max_iter = 1000)

# 2. Train it on your vectorized training data
model.fit(X_train_vec, y_train)

# 3. Use it to predict on the test data
y_pred = model.predict(X_test_vec)