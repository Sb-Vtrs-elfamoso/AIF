import streamlit as st 
import pandas as pd 
from annoy import AnnoyIndex
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re

import pickle
from tokenizer_utils import StemTokenizer
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np 

from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
def main():

    if 'column' not in st.session_state :
        st.session_state['column'] = ""
        st.session_state['df'] = ""
    option = st.selectbox(
        'What type of vectorizer do you want to use ?',
        ('TF-IDF','Gensim'))
    text_input = st.text_input('movie name : ')
    if text_input == '' : 
        st.warning('Please enter a movie name ')
    else :
    

        df = pd.read_csv('out1.csv',low_memory=False)
        if option == 'TF-IDF':
            column = 'tfidf_features'
            index_file = 'movies_tfidf.ann'
            dim = 500  
            annoy_index = AnnoyIndex(dim, 'angular')
            annoy_index.load(index_file)
            stop_words = set(stopwords.words('english'))
    
            with open('tokenizer.pkl','rb') as f:
                tokenizer = pickle.load(f)
            token_stop = tokenizer(' '.join(stop_words))
            print('tokenizer is ready')
            with open('vectorizer.pkl','rb') as f :
                vectorizer = pickle.load(f)
            print('vectorizer is ready')

        else :
            column = 'mean_embeddings'
            index_file = 'movies_gensim.ann'
            dim = 100  
            annoy_index = AnnoyIndex(dim, 'angular')
            annoy_index.load(index_file)
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
        st.session_state['column'] = column
        st.session_state['df'] = df
        
        


        def get_similar_movies(movie_name, num_neighbors=5,df=df, annoy_index= None):
            # Assuming movie names are unique
            movie_description = movie_name
            text_data = [movie_description]
            if option == 'TF-IDF':
                movie_vector = vectorizer.transform(text_data).toarray()[0]
            else :
                movie_vector = compute_mean_embeddings(movie_description, model, words_list)

            similar_indices = annoy_index.get_nns_by_vector(movie_vector, num_neighbors )  # Add 1 to include the movie itself
            similar_movies = df.loc[similar_indices]
            return similar_movies
        movie_name = text_input
        similar_movies = get_similar_movies(movie_name,annoy_index=annoy_index)
        print(similar_movies)

        with st.spinner('Wait for it...'):
            try :

                similar_movies = get_similar_movies(movie_name,annoy_index=annoy_index)

                st.header("Similar movies to "+ movie_name+ " are :")
                k = 0
                for movie in similar_movies['original_title'] :
                    k += 1
                    st.subheader(str(k) + ' ' + movie)
                    
                
            except : 
                st.warning('The movie does not exist')
        

main()
