
# Reference: https://github.com/keisks/robsut-wrod-reocginiton
# Article: http://cs.jhu.edu/~kevinduh/papers/sakaguchi17robsut.pdf

import numpy as np
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder

from . import SpellCorrector
from .binarize import default_alph, default_specialsignals
from shorttext.utils import classification_exceptions as ce
from shorttext.utils import tokenize
from .binarize import SpellingToConcatCharVecEncoder, SCRNNBinarizer


class SCRNNSpellCorrector(SpellCorrector):
    def __init__(self, operation, alph=default_alph, specialsignals=default_specialsignals, concatcharvec_encoder=None):
        self.operation = operation
        self.binarizer = SCRNNBinarizer(alph, specialsignals)
        self.concatcharvec_encoder = SpellingToConcatCharVecEncoder() if concatcharvec_encoder==None else concatcharvec_encoder
        self.onehotencoder = OneHotEncoder()
        self.trained = False

    def preprocess_text_train(self, text):
        if not self.trained:
            raise ce.ModelNotTrainedException()

        for token in tokenize(text):
            if self.operation.upper().startswith('NOISE'):
                xvec = self.binarizer.noise_char(token, self.operation.upper()[6:])
            elif self.operation.upper().startswith('JUMBLE'):
                xvec = self.binarizer.jumble_char(token, self.operation.upper()[7:])
            normtoken = token if self.dictionary.token2id.has_key(token) else '<unk>'
            yvec = self.onehotencoder.transform([self.dictionary[normtoken]]).reshape((len(self.dictionary), 1))
            yield xvec, yvec

    def train(self, text):
        self.dictionary = Dictionary(text+' <unk> <eos>')
        self.onehotencoder(np.arange(len(self.dictionary)).reshape((len(self.dictionary), 1)))
        xylist = [(xvec, yvec) for xvec, yvec in self.preprocess_text_train(text)]
        xtrain = np.array(map(lambda item: item[0], xylist))
        ytrain = np.array(map(lambda item: item[1], xylist))

        # neural network here

    def correct(self, word):
        pass
