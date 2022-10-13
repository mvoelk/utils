
import numpy as np


def calc_similarity_matrix(z):
    # cosine similarity between a batch of vectors
    # z: shape (num_samples, num_features)
    zn = np.linalg.norm(z, axis=-1)
    return (z @ z.T) / (zn[:,None] @ zn[None,:])

def calc_distance_matrix(z):
    # euclidean distance between a batch of vectors
    # z: shape (num_samples, num_features)
    return np.linalg.norm(z[:,None,:] - z[None,:,:], axis=-1)

