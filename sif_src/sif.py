import numpy as np
from collections import Counter

def compute_word_frequencies(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    total_words = sum(word_counts.values())
    word_freq = {word: count / total_words for word, count in word_counts.items()}
    return word_freq

def compute_sif_weights(word_freq, a=1e-3):
    sif_weights = {word: a / (a + freq) for word, freq in word_freq.items()}
    return sif_weights

def compute_sif_embeddings(corpus, word_vectors, sif_weights):
    embeddings = []
    for sentence in corpus:
        weights = np.array([sif_weights.get(word, 0) for word in sentence])
        vectors = np.array([word_vectors[word] for word in sentence if word in word_vectors])
        weighted_avg = np.average(vectors, axis=0, weights=weights)
        embeddings.append(weighted_avg)
    return np.array(embeddings)

def remove_pc(embeddings, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(embeddings)
    pc = svd.components_
    if npc == 1:
        embeddings = embeddings - embeddings.dot(pc.transpose()) * pc
    else:
        embeddings = embeddings - embeddings.dot(pc.transpose()).dot(pc)
    return embeddings
