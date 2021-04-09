
import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
import requests

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
    return KeyedVectors.load_word2vec_format(path, binary=binary)


def load_fasttext_model(path, encoding='utf-8'):
    """ Load a pre-trained FastText model.

    :param path: path of the file of the pre-trained FastText model
    :return: a pre-trained FastText model
    :type path: str
    :rtype: gensim.models.keyedvectors.FastTextKeyedVectors
    """
    return gensim.models.fasttext.load_facebook_vectors(path, encoding=encoding)


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


class RESTfulKeyedVectors(KeyedVectors):
    """ RESTfulKeyedVectors, for connecting to the API of the preloaded word-embedding vectors loaded
        by `WordEmbedAPI`.

        This class inherits from :class:`gensim.models.keyedvectors.KeyedVectors`.

    """
    def __init__(self, url, port='5000'):
        """ Initialize the class.

        :param url: URL of the API, usually `http://localhost`
        :param port: Port number
        :type url: str
        :type port: str
        """
        self.url = url
        self.port = port

    def closer_than(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: list of words
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/closerthan',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()

    def distance(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: distance between two words
        :rtype: float
        """
        r = requests.post(self.url + ':' + self.port + '/distance',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['distance']

    def distances(self, entity1, other_entities=()):
        """

        :param entity1: word
        :param other_entities: list of words
        :type entity1: str
        :type other_entities: list
        :return: list of distances between `entity1` and each word in `other_entities`
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/distances',
                          json={'entity1': entity1, 'other_entities': other_entities})
        return np.array(r.json()['distances'], dtype=np.float32)

    def get_vector(self, entity):
        """

        :param entity: word
        :type: str
        :return: word vectors of the given word
        :rtype: numpy.ndarray
        """
        r = requests.post(self.url + ':' + self.port + '/get_vector', json={'token': entity})
        returned_dict = r.json()
        if 'vector' in returned_dict:
            return np.array(returned_dict['vector'])
        else:
            raise KeyError('The token {} does not exist in the model.'.format(entity))

    def most_similar(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar', json=kwargs)
        return [tuple(pair) for pair in r.json()]

    def most_similar_to_given(self, entity1, entities_list):
        """

        :param entity1: word
        :param entities_list: list of words
        :type entity1: str
        :type entities_list: list
        :return: list of similarities between the given word and each word in `entities_list`
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar_to_given',
                          json={'entity1': entity1, 'entities_list': entities_list})
        return r.json()['token']

    def rank(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: rank
        :rtype: int
        """
        r = requests.post(self.url + ':' + self.port + '/rank',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['rank']

    def save(self, fname_or_handle, **kwargs):
        """

        :param fname_or_handle:
        :param kwargs:
        :return:
        """
        raise IOError('The class RESTfulKeyedVectors do not persist models to a file.')

    def similarity(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :return: similarity between two words
        :type entity1: str
        :type entity2: str
        :rtype: float
        """
        r = requests.post(self.url + ':' + self.port + '/similarity',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['similarity']

# reference: https://radimrehurek.com/gensim/models/keyedvectors.html
