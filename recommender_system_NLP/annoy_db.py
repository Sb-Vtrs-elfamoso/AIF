import numpy as np
from annoy import AnnoyIndex
import pandas as pd 
import ast

def load_data(features_path='train_embeddings.csv'):
    """Load features and paths from disk."""
    features = pd.read_csv(features_path)
    return features

def build_and_save_annoy_index_tfidf(features, n_trees=10, index_file_name='movies_tfidf.ann'):
    """Build and save an Annoy index."""
    dim = 500  # Dimensionality of the features
    annoy_index = AnnoyIndex(dim, 'angular')
    vectors = features['tfidf_features']
    for i, vector in enumerate(vectors):
        numbers_list = ast.literal_eval(vector)
        float_list = [float(num) for num in numbers_list]
        annoy_index.add_item(i, float_list)

    annoy_index.build(n_trees)
    annoy_index.save(index_file_name)
    print(f"Annoy index built with {n_trees} trees and saved to {index_file_name}.")

def build_and_save_annoy_index_gensim(features, n_trees=10, index_file_name='movies_gensim.ann'):
    """Build and save an Annoy index."""
    dim = 100  # Dimensionality of the features
    annoy_index = AnnoyIndex(dim, 'angular')
    vectors = features['mean_embeddings']
    for i, vector in enumerate(vectors):
        split_data = vector.strip('[]').split()
        float_list = [float(x) for x in split_data]
        annoy_index.add_item(i, float_list)

    annoy_index.build(n_trees)
    annoy_index.save(index_file_name)
    print(f"Annoy index built with {n_trees} trees and saved to {index_file_name}.")
def main():
    features = load_data()
    build_and_save_annoy_index_tfidf(features)
    build_and_save_annoy_index_gensim(features)

if __name__ == "__main__":
    main()
