import pandas as pd 
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from tqdm import tqdm
import numpy as np 
# Download stopwords list

traindf = pd.read_csv('out.csv')
print(traindf)


# glove 


glove_file = ('glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)


tqdm.pandas()
def compute_mean_embeddings(s, model, words_list, dim=100):
  s = s.lower()
  emb_list = [model[w] for w in s if w in words_list]
  if emb_list != []:
    return np.mean(emb_list, axis=0)
  else:
    return np.zeros(dim)
  
words_list = model.index_to_key
traindf['mean_embeddings'] = traindf.overview.progress_apply(lambda s: compute_mean_embeddings(s, model, words_list))
  
traindf.to_csv('train_embeddings.csv',index=True)