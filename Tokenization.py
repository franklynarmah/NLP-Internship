#Practicing the tokenization materials

import nltk
from nltk.stem import   PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer





#nltk.download()
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download("punk_tab")
#nltk.download('LazyCorpusLoader')
#nltk.download('PorterStemmer')
#nltk.download('averaged_perceptron_tagger_eng')


#from nltk.tokenize import sent_tokenize
#text = "This assignment analyses a Service Call Record form from Domestic Appliances Warranty Services. The form documents a call made by a service engineer to repair a domestic appliance. Using relational data analysis, the data is progressively normalised from its unnormalised form (UNF) through First, Second, and Third Normal Form (1NF, 2NF, 3NF)."
#tokenized_version_s = sent_tokenize(text)
#print(tokenized_version_s)

#from nltk.tokenize import word_tokenize
sentence = " This assignment analyses a Service Call Record form from Domestic Appliances Warranty Services."
#tokenized_sentence = word_tokenize(sentence)
#print(tokenized_sentence)

#from nltk.corpus import stopwords
#stop_words=set(stopwords.words("english"))
#print(stop_words)

# Stemming
filtered_sent = nltk.word_tokenize(sentence)
PS = PorterStemmer()
stemmed_words=[]
for word in filtered_sent:
    stemmed_words.append(PS.stem(word))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#Lemmatization
lem = WordNetLemmatizer()

word = "was"
Lemmatized_word = lem.lemmatize(word, "v")

print("lemmatized word: ", Lemmatized_word)

#POS tagging
sentence2 = "This assignment analyses a Service Call Record form from Domestic Appliances Warranty Services."
tokenized_version = nltk.word_tokenize(sentence2)
print(tokenized_version)
tagged_version = nltk.pos_tag(tokenized_version)
print(tagged_version)