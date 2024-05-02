import streamlit as st 
import pandas as pd 
from annoy import AnnoyIndex

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
        else :
            column = 'mean_embeddings'
            index_file = 'movies_gensim.ann'
            dim = 100  
            annoy_index = AnnoyIndex(dim, 'angular')
            annoy_index.load(index_file)
        st.session_state['column'] = column
        st.session_state['df'] = df
        
        


        def get_similar_movies(movie_name, num_neighbors=5,df=df, annoy_index= None):
            # Assuming movie names are unique
            idx = df.index[df['original_title'] == movie_name][0]
            similar_indices = annoy_index.get_nns_by_item(idx, num_neighbors + 1)  # Add 1 to include the movie itself
            similar_movies = df.loc[similar_indices].drop(idx)  # Exclude the movie itself
            return similar_movies
        movie_name = text_input
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