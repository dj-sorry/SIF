from sklearn.metrics.pairwise import cosine_similarity

def evaluate_embeddings(embeddings, labels):
    #consider other methods 
    similarities = cosine_similarity(embeddings)
    #not implemented
    return similarities
