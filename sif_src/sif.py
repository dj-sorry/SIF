from typing import Iterable
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from sklearn.preprocessing import normalize


def flatten(lst):
    """
    Flatten a nested list structure.
    
    Parameters:
    lst (list): A potentially nested list of elements.
    
    Yields:
    elements of the nested list in a flattened structure.
    """
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(item)
        else:
            yield item

def compute_word_frequencies(sentences):
    """
    Compute word frequencies from a nested list of sentences.
    
    Parameters:
    sentences (list of list of str): A nested list where each sublist contains sentences.
    
    Returns:
    dict: A dictionary with words as keys and their frequencies as values.
    """
    word_freq = defaultdict(int)
    flattened_sentences = list(flatten(sentences))
    for sentence in flattened_sentences:
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence
        for word in words:
            word_freq[word] += 1
    return word_freq

def compute_sif_weights(word_freq, a=1e-3):
    """
    Compute Smooth Inverse Frequency (SIF) weights for words.
    
    Parameters:
    word_freq (dict): A dictionary with words as keys and their frequencies as values.
    a (float): A smoothing parameter.
    
    Returns:
    dict: A dictionary with words as keys and their SIF weights as values.
    """
    return {word: a / (a + freq) for word, freq in word_freq.items()}

def compute_sif_embeddings_queries(corpus, word_vectors, sif_weights):
    embeddings = []
    for sublist in corpus:
        for sentence in sublist:
            words = sentence.split()
            vectors = []
            weights = []
            for word in words:
                if word in word_vectors and word in sif_weights:
                    vectors.append(word_vectors[word])
                    weights.append(sif_weights[word])
            if vectors:
                vectors = np.array(vectors)
                weights = np.array(weights)
                weighted_avg = np.average(vectors, axis=0, weights=weights)
                embeddings.append(weighted_avg)
            else:
                embeddings.append(np.zeros(next(iter(word_vectors.values())).shape))
    return embeddings


def compute_sif_embeddings_texts(corpus, word_vectors, sif_weights):
    """
    Compute Sentence Embeddings using Smooth Inverse Frequency (SIF) weighting.
    
    Parameters:
    corpus (list of list of str): A nested list where each sublist contains texts.
    word_vectors (dict): A dictionary with words as keys and their vector representations as values.
    sif_weights (dict): A dictionary with words as keys and their SIF weights as values.
    
    Returns:
    list of list of np.ndarray: A list containing embeddings for each sublist of texts.
    """
    embeddings = []
    for texts in corpus:
        text_embeddings = []
        for text in texts:
            words = text.split()
            vectors = []
            weights = []
            for word in words:
                if word in word_vectors and word in sif_weights:
                    vectors.append(word_vectors[word])
                    weights.append(sif_weights[word])
            if vectors:
                vectors = np.array(vectors)
                weights = np.array(weights)
                weighted_avg = np.average(vectors, axis=0, weights=weights)
                text_embeddings.append(weighted_avg)
            else:
                text_embeddings.append(np.zeros(next(iter(word_vectors.values())).shape))
        embeddings.append(text_embeddings)
    return embeddings

def remove_pc_sif(embeddings, n=1, alpha=0.0001):
    """
    Remove the first n principal components from each embedding using the SIF method.

    Parameters:
        embeddings (list of arrays): List of sentence embeddings.
        n (int): Number of principal components to remove.
        alpha (float): Weighting parameter for the removal of principal components.

    Returns:
        embeddings_pc_removed (list of arrays): List of sentence embeddings with principal components removed.
    """
    # Combine all embeddings into a single matrix
    combined_embeddings = np.concatenate(embeddings)

    # Compute the principal components across the entire dataset
    svd = TruncatedSVD(n_components=n, n_iter=7, random_state=0)
    svd.fit(combined_embeddings)

    # Compute the projection of each embedding onto the principal components and remove it
    embeddings_pc_removed = []
    for embedding in embeddings:
        embedding_proj = np.dot(embedding, svd.components_.T)
        embedding_pc_removed = embedding - alpha * embedding_proj
        embeddings_pc_removed.append(embedding_pc_removed)

    # Normalize the embeddings
    embeddings_pc_removed = [normalize(embedding_pc_removed) for embedding_pc_removed in embeddings_pc_removed]

    return embeddings_pc_removed


