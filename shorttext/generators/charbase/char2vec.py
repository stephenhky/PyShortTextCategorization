
from functools import partial

import numpy as np
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder

from shorttext.utils.misc import textfile_generator


class SentenceToCharVecEncoder:
    """ A class that facilitates one-hot encoding from characters to vectors.

    """
    def __init__(self, dictionary, signalchar='\n'):
        """ Initialize the one-hot encoding class.

        :param dictionary: a gensim dictionary
        :param signalchar: signal character, useful for seq2seq models (Default: '\n')
        :type dictionary: gensim.corpora.Dictionary
        :type signalchar: str
        """
        self.dictionary = dictionary
        self.signalchar = signalchar
        numchars = len(self.dictionary)
        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(np.arange(numchars).reshape((numchars, 1)))

    def calculate_prelim_vec(self, sent):
        """ Convert the sentence to a one-hot vector.

        :param sent: sentence
        :return: a one-hot vector, with each element the code of that character
        :type sent: str
        :rtype: numpy.array
        """
        return self.onehot_encoder.transform(
            np.array([self.dictionary.token2id[c] for c in sent]).reshape((len(sent), 1))
        )

    def encode_sentence(self, sent, maxlen, startsig=False, endsig=False):
        """ Encode one sentence to a sparse matrix, with each row the expanded vector of each character.

        :param sent: sentence
        :param maxlen: maximum length of the sentence
        :param startsig: signal character at the beginning of the sentence (Default: False)
        :param endsig: signal character at the end of the sentence (Default: False)
        :return: matrix representing the sentence
        :type sent: str
        :type maxlen: int
        :type startsig: bool
        :type endsig: bool
        :rtype: scipy.sparse.csc_matrix
        """
        cor_sent = (self.signalchar if startsig else '') + sent[:min(maxlen, len(sent))] + (self.signalchar if endsig else '')
        sent_vec = self.calculate_prelim_vec(cor_sent).tocsc()
        if sent_vec.shape[0] == maxlen + startsig + endsig:
            return sent_vec
        else:
            return csc_matrix((sent_vec.data, sent_vec.indices, sent_vec.indptr),
                              shape=(maxlen + startsig + endsig, sent_vec.shape[1]),
                              dtype=np.float64)

    def encode_sentences(self, sentences, maxlen, sparse=True, startsig=False, endsig=False):
        """ Encode many sentences into a rank-3 tensor.

        :param sentences: sentences
        :param maxlen: maximum length of one sentence
        :param sparse: whether to return a sparse matrix (Default: True)
        :param startsig: signal character at the beginning of the sentence (Default: False)
        :param endsig: signal character at the end of the sentence (Default: False)
        :return: rank-3 tensor of the sentences
        :type sentences: list
        :type maxlen: int
        :type sparse: bool
        :type startsig: bool
        :type endsig: bool
        :rtype: scipy.sparse.csc_matrix or numpy.array
        """
        encode_sent_func = partial(self.encode_sentence, startsig=startsig, endsig=endsig, maxlen=maxlen)
        list_encoded_sentences_map = map(encode_sent_func, sentences)
        if sparse:
            return list(list_encoded_sentences_map)
        else:
            return np.array([sparsevec.toarray() for sparsevec in list_encoded_sentences_map])

    def __len__(self):
        return len(self.dictionary)


def initSentenceToCharVecEncoder(textfile, encoding=None):
    """ Instantiate a class of SentenceToCharVecEncoder from a text file.

    :param textfile: text file
    :param encoding: encoding of the text file (Default: None)
    :return: an instance of SentenceToCharVecEncoder
    :type textfile: file
    :type encoding: str
    :rtype: SentenceToCharVecEncoder
    """
    dictionary = Dictionary(map(lambda line: [c for c in line], textfile_generator(textfile, encoding=encoding)))
    return SentenceToCharVecEncoder(dictionary)
