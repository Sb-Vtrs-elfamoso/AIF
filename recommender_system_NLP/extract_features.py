import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re


nltk.download('punkt')
nltk.download('stopwords')

data_path = 'data_NLP/archive/movies_metadata.csv'
df = pd.read_csv(data_path)
print("#### read the dataframe ####")
print(df.head())
print("#### read the overview column ####")
print(df['overview'])

#null values 
print(df["overview"].isna().sum())
clean_df = df.dropna(subset=["overview"])

# analyser


stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", "'"]
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

tokenizer=StemTokenizer()
token_stop = tokenizer(' '.join(stop_words))

# count vectorizer
vectorizer = CountVectorizer(stop_words=token_stop, tokenizer=tokenizer,max_features=500,ngram_range=(1,2))
train_count_matrix = vectorizer.fit_transform(clean_df.overview)
print(train_count_matrix.shape)

# Convert the count matrix into a DataFrame with column names
tfidf_array = train_count_matrix.toarray()
tfidf_lists = [list(row) for row in tfidf_array]
print(type(tfidf_lists))
clean_df['tfidf_features'] = tfidf_lists

print(type(clean_df['tfidf_features']))
clean_df.to_csv('out1.csv')  
print(clean_df)
df = pd.read_csv('out1.csv',low_memory=False)