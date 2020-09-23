
import json

import numpy as np
import gensim

from .s2skeras import Seq2SeqWithKeras, loadSeq2SeqWithKeras, kerasseq2seq_suffices
from ..charbase.char2vec import SentenceToCharVecEncoder
from shorttext.utils import compactmodel_io as cio


charbases2s_suffices = kerasseq2seq_suffices + ['_dictionary.dict', '_charbases2s.json']


class CharBasedSeq2SeqGenerator(cio.CompactIOMachine):
    """ Class implementing character-based sequence-to-sequence (seq2seq) learning model.

    This class implements the seq2seq model at the character level. This class calls
    :class:`Seq2SeqWithKeras`.

    Reference:

    Oriol Vinyals, Quoc Le, "A Neural Conversational Model," arXiv:1506.05869 (2015). [`arXiv
    <https://arxiv.org/abs/1506.05869>`_]
    """
    def __init__(self, sent2charvec_encoder, latent_dim, maxlen):
        """ Instantiate the class.

        If no one-hot encoder passed in, no compilation will be performed.

        :param sent2charvec_encoder: the one-hot encoder
        :param latent_dim: number of latent dimension
        :param maxlen: maximum length of a sentence
        :type sent2charvec_encoder: SentenceToCharVecEncoder
        :type latent_dim: int
        :type maxlen: int
        """
        cio.CompactIOMachine.__init__(self, {'classifier': 'charbases2s'}, 'charbases2s', charbases2s_suffices)
        self.compiled = False
        if sent2charvec_encoder != None:
            self.sent2charvec_encoder = sent2charvec_encoder
            self.dictionary = self.sent2charvec_encoder.dictionary
            self.nbelem = len(self.dictionary)
            self.latent_dim = latent_dim
            self.maxlen = maxlen
            self.s2sgenerator = Seq2SeqWithKeras(self.nbelem, self.latent_dim)

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Compile the keras model.

        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :return: None
        :type optimizer: str
        :type loss: str
        """
        if not self.compiled:
            self.s2sgenerator.prepare_model()
            self.s2sgenerator.compile(optimizer=optimizer, loss=loss)
            self.compiled = True

    def prepare_trainingdata(self, txtseq):
        """ Transforming sentence to a sequence of numerical vectors.

        :param txtseq: text
        :return: rank-3 tensors for encoder input, decoder input, and decoder output
        :type txtseq: str
        :rtype: (numpy.array, numpy.array, numpy.array)
        """
        encoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[:-1], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[1:], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_output = self.sent2charvec_encoder.encode_sentences(txtseq[1:], endsig=True, maxlen=self.maxlen, sparse=False)
        return encoder_input, decoder_input, decoder_output

    def train(self, txtseq, batch_size=64, epochs=100, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Train the character-based seq2seq model.

        :param txtseq: text
        :param batch_size: batch size (Default: 64)
        :param epochs: number of epochs (Default: 100)
        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :return: None
        :type txtseq: str
        :type batch_size: int
        :type epochs: int
        :type optimizer: str
        :type loss: str
        """
        encoder_input, decoder_input, decoder_output = self.prepare_trainingdata(txtseq)
        self.compile(optimizer=optimizer, loss=loss)
        self.s2sgenerator.fit(encoder_input, decoder_input, decoder_output, batch_size=batch_size, epochs=epochs)

    def decode(self, txtseq, stochastic=True):
        """ Given an input text, produce the output text.

        :param txtseq: input text
        :return: output text
        :type txtseq: str
        :rtype: str
        """
        # Encode the input as state vectors.
        inputvec = np.array([self.sent2charvec_encoder.encode_sentence(txtseq, maxlen=self.maxlen, endsig=True).toarray()])
        states_value = self.s2sgenerator.encoder_model.predict(inputvec)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.nbelem))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.dictionary.token2id['\n']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_txtseq = ''
        while not stop_condition:
            output_tokens, h, c = self.s2sgenerator.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            if stochastic:
                sampled_token_index = np.random.choice(np.arange(output_tokens.shape[2]),
                                                       p=output_tokens[0, -1, :])
            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.dictionary[sampled_token_index]
            decoded_txtseq += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_txtseq) > self.maxlen):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.nbelem))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_txtseq

    def savemodel(self, prefix, final=False):
        """ Save the trained models into multiple files.

        To save it compactly, call :func:`~save_compact_model`.

        If `final` is set to `True`, the model cannot be further trained.

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param prefix: prefix of the file path
        :param final: whether the model is final (that should not be trained further) (Default: False)
        :return: None
        :type prefix: str
        :type final: bool
        :raise: ModelNotTrainedException
        """
        self.s2sgenerator.savemodel(prefix, final=final)
        self.dictionary.save(prefix+'_dictionary.dict')
        json.dump({'maxlen': self.maxlen, 'latent_dim': self.latent_dim}, open(prefix+'_charbases2s.json', 'w'))

    def loadmodel(self, prefix):
        """ Load a trained model from various files.

        To load a compact model, call :func:`~load_compact_model`.

        :param prefix: prefix of the file path
        :return: None
        :type prefix: str
        """
        self.dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
        self.s2sgenerator = loadSeq2SeqWithKeras(prefix, compact=False)
        self.sent2charvec_encoder = SentenceToCharVecEncoder(self.dictionary)
        self.nbelem = len(self.dictionary)
        hyperparameters = json.load(open(prefix+'_charbases2s.json', 'r'))
        self.latent_dim, self.maxlen = hyperparameters['latent_dim'], hyperparameters['maxlen']
        self.compiled = True

def loadCharBasedSeq2SeqGenerator(path, compact=True):
    """ Load a trained `CharBasedSeq2SeqGenerator` class from file.

    :param path: path of the model file
    :param compact: whether it is a compact model (Default: True)
    :return: a `CharBasedSeq2SeqGenerator` class for sequence to sequence inference
    :type path: str
    :type compact: bool
    :rtype: CharBasedSeq2SeqGenerator
    """
    seq2seqer = CharBasedSeq2SeqGenerator(None, 0, 0)
    if compact:
        seq2seqer.load_compact_model(path)
    else:
        seq2seqer.loadmodel(path)
    return seq2seqer