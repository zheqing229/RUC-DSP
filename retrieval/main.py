import numpy as np
from dataload import load_dataset
from evalute import match_query,evaluate

def main(data_dir, query_fold=5, n_mfcc=20, frame_length=2048, hop_length=512, top_k=20):

    database, query_set = load_dataset(file_dir=data_dir, query_fold=query_fold, n_mfcc=20,  window_length=frame_length, hop_size=hop_length)
    database_features = np.array([item[0] for item in database])
    database_labels = [item[1] for item in database]
    query_features = np.array([item[0] for item in query_set])
    query_labels = [item[1] for item in query_set]
    top_labels = match_query(query_features, database_features, database_labels, k=top_k)
    

    top_10_acc = evaluate(query_labels, top_labels, k=10)
    top_20_acc = evaluate(query_labels, top_labels, k=20)
    
    print(f"Top-10 Accuracy: {top_10_acc * 100:.2f}%")
    print(f"Top-20 Accuracy: {top_20_acc * 100:.2f}%")


if __name__ == "__main__":
    DATA_DIR = "audio"  
    QUERY_FOLD = 5
    FRAME_LENGTH = 2048  
    HOP_LENGTH = 512    
    TOP_K = 20
    N_MFCC = 20
    main(DATA_DIR, QUERY_FOLD, N_MFCC, FRAME_LENGTH, HOP_LENGTH, TOP_K)
