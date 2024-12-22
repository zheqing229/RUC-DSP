import numpy as np
from scipy.spatial.distance import cosine
from numpy.linalg import norm

def normalized_cross_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.sum((a_flat - np.mean(a_flat)) * (b_flat - np.mean(b_flat))) / (
        np.sqrt(np.sum((a_flat - np.mean(a_flat)) ** 2)) * np.sqrt(np.sum((b_flat - np.mean(b_flat)) ** 2))
    )

def cosine_similarity(a, b):
    a_flat = a.flatten()  # 将矩阵展平为向量
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (norm(a_flat) * norm(b_flat))

def match_query(query_features, database_features, database_labels, k=10):
    '''
    Matches query features with the database features and returns the top-k labels based on cosine distance.

    query_features: A numpy array of features for the query set (shape: [n_queries, feature_dim, time_steps]).
    database_features: A numpy array of features for the database set (shape: [n_database, feature_dim, time_steps]).
    database_labels: A list or numpy array of labels for the database set.
    k: The number of top matches to return (default is 10).

    Returns:
    - top_labels: A numpy array of shape [n_queries, k], where each row contains the top-k labels for a query.
    '''
    
    distances = np.zeros((query_features.shape[0], database_features.shape[0]))

    for i, query in enumerate(query_features):
        for j, db_feature in enumerate(database_features):
            distances[i, j] = normalized_cross_correlation(query, db_feature) 
            # distances = cosine_similarity(query_features, database_features)

    top_indices = np.argsort(-distances, axis=1)[:, :k] 
   
    top_labels = np.array(database_labels)[top_indices]
    return top_labels

def evaluate(query_labels, top_labels, k):
    '''
    Evaluates the accuracy of the top-k labels for each query.

    query_labels: A list or numpy array of true labels for the query set.
    top_labels: A numpy array of shape [n_queries, k], containing the top-k predicted labels for each query.
    k: The number of top labels to consider for evaluation.

    Returns:
    - accuracy: The proportion of correct labels in the top-k predictions.
    '''
    correct = 0
    for i in range(len(query_labels)):
        correct += np.sum(top_labels[i, :k] == query_labels[i])
    accuracy = correct / (len(query_labels) *k)
    return accuracy