
import numpy as np
import gensim

from shorttext.utils import tokenize

def load_word2vec_model(path, binary=True):
    """ Load a pre-trained Word2Vec model.

    :param path: path of the file of the pre-trained Word2Vec model
    :param binary: whether the file is in binary format (Default: True)
    :return: a pre-trained Word2Vec model
    :type path: str
    :type binary: bool
    :rtype: gensim.models.keyedvectors.KeyedVectors
    """
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)

def shorttext_to_avgembedvec(shorttext, wvmodel, vecsize):
    """ Convert the short text into an averaged embedded vector representation.

    Given a short sentence, it converts all the tokens into embedded vectors according to
    the given word-embedding model, sums
    them up, and normalize the resulting vector. It returns the resulting vector
    that represents this short sentence.

    :param shorttext: a short sentence
    :param wvmodel: word-embedding model
    :param vecsize: length of embedded vector
    :return: an embedded vector that represents the short sentence
    :type shorttext: str
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type vecsize: int
    :rtype: numpy.ndarray
    """
    vec = np.zeros(vecsize)
    for token in tokenize(shorttext):
        if token in wvmodel:
            vec += wvmodel[token]
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec /= np.linalg.norm(vec)
    return vec
