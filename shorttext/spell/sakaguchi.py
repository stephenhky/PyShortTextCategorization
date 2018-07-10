
# Reference: https://github.com/keisks/robsut-wrod-reocginiton
# Article: http://cs.jhu.edu/~kevinduh/papers/sakaguchi17robsut.pdf

import json

import numpy as np
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout, Dense, TimeDistributed

import shorttext.utils.kerasmodel_io as kerasio
from . import SpellCorrector
from .binarize import default_alph, default_specialsignals
from shorttext.utils import classification_exceptions as ce
from .binarize import SpellingToConcatCharVecEncoder, SCRNNBinarizer
from shorttext.utils.compactmodel_io import CompactIOMachine


nospace_tokenize = lambda sentence: [t.strip() for t in sentence.split() if len(t.strip())>0]


class SCRNNSpellCorrector(SpellCorrector, CompactIOMachine):
    """ scRNN (semi-character-level recurrent neural network) Spell Corrector.

    Reference:
    Keisuke Sakaguchi, Kevin Duh, Matt Post, Benjamin Van Durme, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Networ," arXiv:1608.02214 (2016). [`arXiv
    <https://arxiv.org/abs/1608.02214>`_]

    """
    def __init__(self, operation,
                 alph=default_alph,
                 specialsignals=default_specialsignals,
                 concatcharvec_encoder=None,
                 batchsize=1,
                 nb_hiddenunits=650):
        """ Instantiate the scRNN spell corrector.

        :param operation: types of distortion of words in training (options: "NOISE-INSERT", "NOISE-DELETE", "NOISE-REPLACE", "JUMBLE-WHOLE", "JUMBLE-BEG", "JUMBLE-END", and "JUMBLE-INT")
        :param alph: default string of characters (Default: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#")
        :param specialsignals: dictionary of special signals (Default built-in)
        :param concatcharvec_encoder: one-hot encoder for characters, initialize if None. (Default: None)
        :param batchsize: batch size. (Default: 1)
        :param nb_hiddenunits: number of hidden units. (Default: 650)
        :type operation: str
        :type alpha: str
        :type specialsignals: dict
        :type concatcharvec_encoder: shorttext.spell.binarize.SpellingToConcatCharVecEncoder
        :type batchsize: int
        :type nb_hiddenunits: int
        """
        CompactIOMachine.__init__(self, {'classifier': 'scrnn_spell'}, 'scrnn_spell', ['_config.json', '_vocabs.gensimdict', '.h5', '.json'])
        self.operation = operation
        self.alph = alph
        self.specialsignals = specialsignals
        self.binarizer = SCRNNBinarizer(self.alph, self.specialsignals)
        self.concatcharvec_encoder = SpellingToConcatCharVecEncoder(self.alph) if concatcharvec_encoder==None else concatcharvec_encoder
        self.onehotencoder = OneHotEncoder()
        self.trained = False
        self.batchsize = batchsize
        self.nb_hiddenunits = nb_hiddenunits

    def preprocess_text_train(self, text):
        """ A generator that output numpy vectors for the text for training.

        :param text: text
        :return: generator that outputs the numpy vectors for training
        :type text: str
        :rtype: generator
        """
        for token in nospace_tokenize(text):
            if self.operation.upper().startswith('NOISE'):
                xvec, _ = self.binarizer.noise_char(token, self.operation.upper()[6:])
            elif self.operation.upper().startswith('JUMBLE'):
                xvec, _ = self.binarizer.jumble_char(token, self.operation.upper()[7:])
            normtoken = token if token in self.dictionary.token2id else '<unk>'
            yvec = self.onehotencoder.transform([[self.dictionary.token2id[normtoken]]]).toarray().reshape((len(self.dictionary), 1))
            yield xvec, yvec

    def preprocess_text_correct(self, text):
        """ A generator that output numpy vectors for the text for correction.

        ModelNotTrainedException is raised if the model has not been trained.

        :param text: text
        :return: generator that outputs the numpy vectors for correction
        :type text: str
        :rtype: generator
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise ce.ModelNotTrainedException()
        for token in nospace_tokenize(text):
            xvec, _ = self.binarizer.change_nothing(token, self.operation)
            yield xvec

    def train(self, text, nb_epoch=100, dropout_rate=0.01, optimizer='rmsprop'):
        """ Train the scRNN model.

        :param text: training corpus
        :param nb_epoch: number of epochs (Default: 100)
        :param dropout_rate: dropout rate (Default: 0.01)
        :param optimizer: optimizer (Default: "rmsprop")
        :type text: str
        :type nb_epoch: int
        :type dropout_rate: float
        :type optimizer: str
        """
        self.dictionary = Dictionary([nospace_tokenize(text), default_specialsignals.values()])
        self.onehotencoder.fit(np.arange(len(self.dictionary)).reshape((len(self.dictionary), 1)))
        xylist = [(xvec.transpose(), yvec.transpose()) for xvec, yvec in self.preprocess_text_train(text)]
        xtrain = np.array([item[0] for item in xylist])
        ytrain = np.array([item[1] for item in xylist])

        # neural network here
        model = Sequential()
        model.add(LSTM(self.nb_hiddenunits, return_sequences=True, batch_input_shape=(None, self.batchsize, len(self.concatcharvec_encoder)*3)))
        model.add(Dropout(dropout_rate))
        model.add(TimeDistributed(Dense(len(self.dictionary))))
        model.add(Activation('softmax'))

        # compile... more arguments
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # training
        model.fit(xtrain, ytrain, epochs=nb_epoch)

        self.model = model
        self.trained = True

    def correct(self, word):
        """ Recommend a spell correction to given the word.

        :param word: a given word
        :return: recommended correction
        :type word: str
        :rtype: str
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise ce.ModelNotTrainedException()

        xmat = np.array([xvec.transpose() for xvec in self.preprocess_text_correct(word)])
        yvec = self.model.predict(xmat)

        maxy = yvec.argmax(axis=-1)
        return ' '.join([self.dictionary[y] for y in maxy[0]])

    def loadmodel(self, prefix):
        """ Load the model.

        :param prefix: prefix of the model path
        :return: None
        :type prefix: str
        """
        self.dictionary = Dictionary.load(prefix+'_vocabs.gensimdict')
        parameters = json.load(open(prefix+'_config.json', 'r'))
        self.operation = parameters['operation']
        self.alph = parameters['alph']
        self.specialsignals = parameters['special_signals']
        self.binarizer = SCRNNBinarizer(self.alph, self.specialsignals)
        self.concatcharvec_encoder = SpellingToConcatCharVecEncoder(self.alph)
        self.batchsize = parameters['batchsize']
        self.nb_hiddenunits = parameters['nb_hiddenunits']
        self.onehotencoder = OneHotEncoder()
        self.onehotencoder.fit(np.arange(len(self.dictionary)).reshape((len(self.dictionary), 1)))
        self.model = kerasio.load_model(prefix)
        self.trained = True

    def savemodel(self, prefix):
        """ Save the model.

        :param prefix: prefix of the model path
        :return: None
        :type prefix: str
        """
        if not self.trained:
            raise ce.ModelNotTrainedException()
        kerasio.save_model(prefix, self.model)
        self.dictionary.save(prefix+'_vocabs.gensimdict')
        parameters = {'alph': self.alph, 'special_signals': self.specialsignals, 'operation': self.operation,
                      'batchsize': self.batchsize, 'nb_hiddenunits': self.nb_hiddenunits}
        json.dump(parameters, open(prefix+'_config.json', 'w'))


def loadSCRNNSpellCorrector(filepath, compact=True):
    """ Load a pre-trained scRNN spell corrector instance.

    :param filepath: path of the model if compact==True; prefix of the model oath if compact==False
    :param compact: whether model file is compact (Default: True)
    :return: an instance of scRnn spell corrector
    :type filepath: str
    :type compact: bool
    :rtype: SCRNNSpellCorrector
    """
    corrector = SCRNNSpellCorrector('JUMBLE-WHOLE')
    if compact:
        corrector.load_compact_model(filepath)
    else:
        corrector.loadmodel(filepath)
    return corrector