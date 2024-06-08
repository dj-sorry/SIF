import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

def compute_word_frequencies(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence.split())
    total_words = sum(word_counts.values())
    word_freq = {word: count / total_words for word, count in word_counts.items()}
    return word_freq

def compute_sif_weights(word_freq, a=1e-3):
    sif_weights = {word: a / (a + freq) for word, freq in word_freq.items()}
    return sif_weights

def compute_sif_embeddings(corpus, word_vectors, sif_weights):
    embeddings = []
    for sentence in tqdm(corpus, desc="Computing SIF embeddings"):
        words = sentence.split()
        valid_words = [word for word in words if word in word_vectors]
        if not valid_words:
            embeddings.append(np.zeros(word_vectors[list(word_vectors.keys())[0]].shape))
            continue
        weights = np.array([sif_weights.get(word, 0.0) for word in valid_words])
        vectors = np.array([word_vectors[word] for word in valid_words])
        if len(weights) != vectors.shape[0]:
            print(f"Debug: Length of weights {len(weights)}, shape of vectors {vectors.shape}")
            raise ValueError("Length of weights not compatible with specified axis.")
        weighted_avg = np.average(vectors, axis=0, weights=weights)
        embeddings.append(weighted_avg)
    return np.array(embeddings)

def remove_pc(embeddings, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(embeddings)
    pc = svd.components_
    if npc == 1:
        embeddings = embeddings - np.dot(embeddings, pc.T) * pc
    else:
        embeddings = embeddings - np.dot(np.dot(embeddings, pc.T), pc)
    return embeddings