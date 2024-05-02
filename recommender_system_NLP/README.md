# Movie poster NLP Recommender System

## Requirements :
- Docker engine is installed on your computer. If it is not the case, please follow these [instructions](https://docs.docker.com/engine/install/)
- have a stable internet connection (for packages installation)


## User guide :
1. download or clone this respository
2. go inside the folder **recomender_system_NLP** with a terminal
3. unzip train_embeddings.csv file containing the TF-IDF and Gensim embeddings.
3. run : `docker build -t streamlit-app .` and then run 'docker run -p 8501:8501 streamlit-app'
4. in your brower open the [web-app](http://localhost:8501/)
5. you can now choose the embedding technique, enter any movie_name and submit it, the system will recommend you 5 similar movie names
6. to close all the containers and remove their images run : `sudo docker-compose down --rmi all`
if you want to keep the images run : `sudo docker-compose down`

## Remarks 
the file movies_tfidf.ann contains the annoy indexer for the tfidf vectors.
the file movies_gensim_tfifd.ann contains the annoy indexer for the gensim vectors.
These two files were created using annoy_db.py