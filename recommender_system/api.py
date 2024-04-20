from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import numpy as np

app = Flask(__name__)

# Load the Annoy index
index_file = 'movies.ann'
dim = 576  # Dimensionality of the feature vectors, adjust if necessary
annoy_index = AnnoyIndex(dim, 'angular')
annoy_index.load(index_file)

# Load the features
features = np.load('features.npy', allow_pickle=True)


@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!'


@app.route('/query', methods=['POST'])
def query_index():
    # Extract the movie index from the request JSON
    movie_feature = request.json['feature']
    
    # Retrieve the feature vector using the provided index
   
    feature_vector = movie_feature
    
    # Query the Annoy index
    n_neighbors = 6  # The number of similar movies to find
    nearest_ids = annoy_index.get_nns_by_vector(feature_vector, n_neighbors)
    nearest_ids = nearest_ids[1:]
    # For simplicity, returning the indices of the nearest neighbors
    # You might want to map these back to movie names or paths
    return jsonify(nearest_ids)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)

