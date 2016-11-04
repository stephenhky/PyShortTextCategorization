import pickle
from collections import defaultdict
from operator import add

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from nltk import word_tokenize
from scipy.spatial.distance import cosine

from ..utils import kerasmodel_io as kerasio
from ..utils import ModelNotTrainedException


# Reference: Francois Chollet, "Building Autoencoders in Keras"
# Link: https://blog.keras.io/building-autoencoders-in-keras.html

class AutoEncoderWord2VecClassifier:
    def __init__(self, wvmodel, classdict=None, vecsize=300, encoding_dim=10):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.vecsize = vecsize
        self.encoding_dim = encoding_dim
        self.trained = False

    def train(self):
        # define all the layers of the autoencoder
        input_vec = Input(shape=(self.vecsize,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_vec)
        decoded = Dense(self.vecsize, activation='sigmoid')(encoded)

        # define the autoencoder model
        autoencoder = Model(input=input_vec, output=decoded)

        # define the encoder
        encoder = Model(input=input_vec, output=encoded)

        # define the decoder
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        # compile the autoencoder
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # process training data
        embedvecs = np.array(reduce(add,
                                    [map(self.shorttext_to_embedvec, self.classdict[classtype])
                                     for classtype in self.classdict]
                                    )
                             )

        # fit the model
        autoencoder.fit(embedvecs, embedvecs,
                        nb_epoch=50,
                        batch_size=256,
                        shuffle=True)

        # store the autoencoder models
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        # calculate the training data
        self.addvec = defaultdict(lambda : np.zeros(self.vecsize))
        for classtype in self.classdict:
            for shorttext in self.classdict[classtype]:
                self.addvec[classtype] += self.shorttext_to_embedvec(shorttext)
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.train_encoded_vecs = {classtype: self.encoder.predict(np.array([embedvec]))
                                   for (classtype, embedvec) in self.addvec.items()}

        # flag setting
        self.trained = True

    def savemodel(self, nameprefix, save_complete_autoencoder=False):
        if not self.trained:
            raise ModelNotTrainedException()
        kerasio.save_model(nameprefix+'_encoder', self.encoder)
        if save_complete_autoencoder:
            kerasio.save_model(nameprefix+'_decoder', self.decoder)
            kerasio.save_model(nameprefix+'_autoencoder', self.autoencoder)
        pickle.dump(self.train_encoded_vecs, open(nameprefix+'_trainedvecs.pickle', 'w'))

    def loadmodel(self, nameprefix):
        self.encoder = kerasio.load_model(nameprefix+'encoder')
        self.train_encoded_vecs = pickle.load(open(nameprefix+'_trainedvecspickle', 'r'))
        self.trained = True

    def shorttext_to_embedvec(self, shorttext):
        vec = np.zeros(self.vecsize)
        tokens = word_tokenize(shorttext)
        for token in tokens:
            if token in self.wvmodel:
                vec += self.wvmodel[token]
        vec /= np.linalg.norm(vec)
        return vec

    def encode(self, shorttext):
        if not self.trained:
            raise ModelNotTrainedException()
        embedvec = self.shorttext_to_embedvec(shorttext)
        return self.encoder.predict(np.array([embedvec]))

    def score(self, shorttext):
        encoded_vec = self.encode(shorttext)
        scoredict = {}
        for classtype in self.train_encoded_vecs:
            try:
                scoredict[classtype] = 1 - cosine(encoded_vec, self.train_encoded_vecs[classtype])
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict