from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import re

class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", "'"]
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]
