
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
import gensim

from shorttext.utils import tokenize, deprecated


def load_word2vec_model(path, binary=True):
    """ Load a pre-trained Word2Vec model.

    :param path: path of the file of the pre-trained Word2Vec model
    :param binary: whether the file is in binary format (Default: True)
    :return: a pre-trained Word2Vec model
    :type path: str
    :type binary: bool
    :rtype: gensim.models.keyedvectors.KeyedVectors
    """
    return KeyedVectors.load_word2vec_format(path, binary=binary)


def load_fasttext_model(path):
    """ Load a pre-trained FastText model.

    :param path: path of the file of the pre-trained FastText model
    :return: a pre-trained FastText model
    :type path: str
    :rtype: gensim.models.keyedvectors.FastTextKeyedVectors
    """
    return gensim.models.FastText.load_facebook_vectors(path)


def load_poincare_model(path, word2vec_format=True, binary=False):
    """ Load a Poincare embedding model.

    :param path: path of the file of the pre-trained Poincare embedding model
    :param word2vec_format: whether to load from word2vec format (default: True)
    :param binary: binary format (default: False)
    :return: a pre-trained Poincare embedding model
    :type path: str
    :type word2vec_format: bool
    :type binary: bool
    :rtype: gensim.models.poincare.PoincareKeyedVectors
    """
    if word2vec_format:
        return PoincareKeyedVectors.load_word2vec_format(path, binary=binary)
    else:
        return PoincareModel.load(path).kv


@deprecated
def shorttext_to_avgembedvec(shorttext, wvmodel, vecsize):
    """ Convert the short text into an averaged embedded vector representation. (deprecated, kept for backward compatibility)

    Given a short sentence, it converts all the tokens into embedded vectors according to
    the given word-embedding model, sums
    them up, and normalize the resulting vector. It returns the resulting vector
    that represents this short sentence.

    This function has been deprecated. Please use :func:`shorttext_to_avgvec` instead.

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


def shorttext_to_avgvec(shorttext, wvmodel):
    """ Convert the short text into an averaged embedded vector representation.

    Given a short sentence, it converts all the tokens into embedded vectors according to
    the given word-embedding model, sums
    them up, and normalize the resulting vector. It returns the resulting vector
    that represents this short sentence.

    :param shorttext: a short sentence
    :param wvmodel: word-embedding model
    :return: an embedded vector that represents the short sentence
    :type shorttext: str
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :rtype: numpy.ndarray
    """
    vec = np.sum([wvmodel[token] for token in tokenize(shorttext) if token in wvmodel], axis=0)

    # normalize
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec /= norm

    return vec
