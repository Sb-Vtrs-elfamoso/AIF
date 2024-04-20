import numpy as np
from annoy import AnnoyIndex

def load_data(features_path='features.npy', paths_path='paths.npy'):
    """Load features and paths from disk."""
    features = np.load(features_path, allow_pickle=True)
    paths = np.load(paths_path, allow_pickle=True)
    return features, paths

def build_and_save_annoy_index(features, n_trees=10, index_file_name='movies.ann'):
    """Build and save an Annoy index."""
    dim = features.shape[1]  # Dimensionality of the features
    annoy_index = AnnoyIndex(dim, 'angular')
    
    for i, vector in enumerate(features):
        annoy_index.add_item(i, vector)

    annoy_index.build(n_trees)
    annoy_index.save(index_file_name)
    print(f"Annoy index built with {n_trees} trees and saved to {index_file_name}.")

def main():
    features, paths = load_data()
    build_and_save_annoy_index(features)

if __name__ == "__main__":
    main()
