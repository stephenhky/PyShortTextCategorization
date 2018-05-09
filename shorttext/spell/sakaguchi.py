
# Reference: https://github.com/keisks/robsut-wrod-reocginiton
# Article: http://cs.jhu.edu/~kevinduh/papers/sakaguchi17robsut.pdf

import numpy as np
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout, Dense, TimeDistributed

from . import SpellCorrector
from .binarize import default_alph, default_specialsignals
from shorttext.utils import classification_exceptions as ce
from shorttext.utils import tokenize
from .binarize import SpellingToConcatCharVecEncoder, SCRNNBinarizer


nospace_tokenize = lambda sentence: map(lambda t: t.strip(), filter(lambda t: len(t.strip())>0, tokenize(sentence)))


class SCRNNSpellCorrector(SpellCorrector):
    def __init__(self, operation,
                 alph=default_alph,
                 specialsignals=default_specialsignals,
                 concatcharvec_encoder=None,
                 batchsize=1,
                 nb_hiddenunits=650):
        self.operation = operation
        self.binarizer = SCRNNBinarizer(alph, specialsignals)
        self.concatcharvec_encoder = SpellingToConcatCharVecEncoder(alph) if concatcharvec_encoder==None else concatcharvec_encoder
        self.onehotencoder = OneHotEncoder()
        self.trained = False
        self.batchsize = batchsize
        self.nb_hiddenunits = nb_hiddenunits

    def preprocess_text_train(self, text):
        for token in nospace_tokenize(text):
            print token, len(token)     # DEBUG!!!!!!!!!!!!!!!!!!!!!!!!
            if self.operation.upper().startswith('NOISE'):
                xvec = self.binarizer.noise_char(token, self.operation.upper()[6:])
            elif self.operation.upper().startswith('JUMBLE'):
                xvec = self.binarizer.jumble_char(token, self.operation.upper()[7:])
            normtoken = token if self.dictionary.token2id.has_key(token) else '<unk>'
            yvec = self.onehotencoder.transform([self.dictionary[normtoken]]).reshape((len(self.dictionary), 1))
            yield xvec, yvec

    def preprocess_text_correct(self, text):
        if not self.trained:
            raise ce.ModelNotTrainedException()
        for token in nospace_tokenize(text):
            xvec = self.binarizer.change_nothing(token, self.operation)
            yield xvec

    def train(self, text, nb_epoch=100):
        self.dictionary = Dictionary([nospace_tokenize(text), default_specialsignals.keys()])
        xylist = [(xvec, yvec) for xvec, yvec in self.preprocess_text_train(text)]
        xtrain = np.array(map(lambda item: item[0], xylist))
        ytrain = np.array(map(lambda item: item[1], xylist))

        # neural network here
        model = Sequential()
        model.add(LSTM(self.nb_hiddenunits, return_sequences=True, batch_input_shape=(None, self.batchsize, len(self.concatcharvec_encoder))))
        model.add(Dropout(0.01))
        model.add(TimeDistributed(Dense(len(self.dictionary))))
        model.add(Activation('softmax'))

        # compile... more arguments
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop'  # or sgd
                      #metrics=['accuracy'])
                      )

        model.fit(xtrain, ytrain, nb_epoch=nb_epoch)

        self.model = model

    def correct(self, word):
        xmat = np.array([xvec for xvec in self.preprocess_text_correct(word)])
        yvec = self.model.predict(xmat)

        maxy = yvec.argmax(axis=-1)
        return ' '.join([self.dictionary[y] for y in maxy])