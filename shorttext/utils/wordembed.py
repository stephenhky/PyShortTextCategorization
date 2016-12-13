import tempfile
import os

from gensim.models import Word2Vec
from gensim.scripts import glove2word2vec

def load_word2vec_model(path, binary=True):
    """ Load a pre-trained Word2Vec model.

    :param path: path of the file of the pre-trained Word2Vec model
    :param binary: whether the file is in binary format (Default: True)
    :return: a pre-trained Word2Vec model
    :type path: str
    :type binary: bool
    :rtype: gensim.models.Word2Vec
    """
    return Word2Vec.load_word2vec_format(path, binary=binary)

# TODO: not working
def load_glove_as_word2vec(path):
    tmpwvmodelfile = tempfile.mkstemp(suffix='.txt')
    glove2word2vec.glove2word2vec(path, tmpwvmodelfile)
    wvmodel = load_word2vec_model(tmpwvmodelfile, binary=False)
    os.remove(tmpwvmodelfile)
    return wvmodel