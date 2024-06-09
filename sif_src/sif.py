from typing import Iterable
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


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

import numpy as np

def compute_sif_embeddings(corpus, word_vectors, sif_weights):
    """
    Compute Sentence Embeddings using Smooth Inverse Frequency (SIF) weighting.
    
    Parameters:
    corpus (list of list of str): A nested list where each sublist contains sentences.
    word_vectors (dict): A dictionary with words as keys and their vector representations as values.
    sif_weights (dict): A dictionary with words as keys and their SIF weights as values.
    
    Returns:
    np.ndarray: An array of sentence embeddings.
    """
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
    return np.array(embeddings)


def remove_pc(embeddings, npc=1):
    """
    Remove the projection on the principal components from embeddings.
    
    Parameters:
    embeddings (np.ndarray): An array of embeddings.
    npc (int): Number of principal components to remove.
    
    Returns:
    np.ndarray: Embeddings with principal components removed.
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(embeddings)
    pc = svd.components_
    if npc == 1:
        embeddings -= np.dot(embeddings, pc.T) * pc
    else:
        embeddings -= np.dot(np.dot(embeddings, pc.T), pc)
    return embeddings