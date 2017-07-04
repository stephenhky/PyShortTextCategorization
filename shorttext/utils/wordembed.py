import gensim

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

