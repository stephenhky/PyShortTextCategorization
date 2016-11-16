import pickle
from collections import defaultdict
from operator import add

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from nltk import word_tokenize
from scipy.spatial.distance import cosine

# from ... import kerasmodel_io as kerasio
# from ... import classification_exceptions as e
import utils.kerasmodel_io as kerasio
import utils.classification_exceptions as e

# Reference: Francois Chollet, "Building Autoencoders in Keras"
# Link: https://blog.keras.io/building-autoencoders-in-keras.html

class AutoEncoderWord2VecClassifier:
    """
    This is a supervised classification algorithm for short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    The classification score is determined by the cosine similarity of the encoded vectors of
    the input text and that of the trained data of that class label.

    A reference about how an autoencoder is written with keras by Francois Chollet, titled
    `Building Autoencoders in Keras
    <https://blog.keras.io/building-autoencoders-in-keras.html>`_ .

    A pre-trained Google Word2Vec model can be downloaded `here
    <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.


    """

    def __init__(self, wvmodel, vecsize=300, encoding_dim=10):
        """ Initialize the classifier.

        :param wvmodel: Word2Vec model
        :param vecsize: length of the embedded vectors in the model (Default: 300)
        :param encoding_dim: encoded dimension
        :type wvmodel: gensim.models.word2vec.Word2Vec
        :type vecsize: int
        :type encoding_dim: int
        """
        self.wvmodel = wvmodel
        self.vecsize = vecsize
        self.encoding_dim = encoding_dim
        self.trained = False

    def train(self, classdict):
        """ Train the classifier.

        If this has not been run, or a model was not loaded by :func:`~loadmodel`,
        a `ModelNotTrainedException` will be raised.

        :param classdict: training data
        :return: None
        :type classdict: dict
        """
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
                                    [map(self.shorttext_to_embedvec, classdict[classtype])
                                     for classtype in classdict]
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
        for classtype in classdict:
            for shorttext in classdict[classtype]:
                self.addvec[classtype] += self.shorttext_to_embedvec(shorttext)
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.train_encoded_vecs = {classtype: self.encoder.predict(np.array([embedvec]))
                                   for (classtype, embedvec) in self.addvec.items()}

        # flag setting
        self.trained = True

    def savemodel(self, nameprefix, save_complete_autoencoder=False):
        """Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There are files with names ending with "_encoder.json" and "_encoder.h5", which are
        the JSON and HDF5 files for the encoder respectively.
        And there is a file with name ending with "_trainedvecs.pickle" storing the names of the class labels.
        If `save_complete_autoencoder` is True,
        then there are also files with names ending with "_decoder.json" and "_decoder.h5".

        If there is no trained model, a `ModelNotTrainedException`will be thrown.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        kerasio.save_model(nameprefix+'_encoder', self.encoder)
        if save_complete_autoencoder:
            kerasio.save_model(nameprefix+'_decoder', self.decoder)
            kerasio.save_model(nameprefix+'_autoencoder', self.autoencoder)
        pickle.dump(self.train_encoded_vecs, open(nameprefix+'_trainedvecs.pickle', 'w'))

    def loadmodel(self, nameprefix):
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_trainedvecs.pickle", "_encoder.json", and "_encoder.h5".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        """
        self.encoder = kerasio.load_model(nameprefix+'encoder')
        self.train_encoded_vecs = pickle.load(open(nameprefix+'_trainedvecspickle', 'r'))
        self.trained = True

    def shorttext_to_embedvec(self, shorttext):
        """ Convert the short text into an averaged embedded vector representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, sums
        them up, and normalize the resulting vector. It returns the resulting vector
        that represents this short sentence.

        :param shorttext: a short sentence
        :return: an embedded vector that represents the short sentence
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        vec = np.zeros(self.vecsize)
        tokens = word_tokenize(shorttext)
        for token in tokens:
            if token in self.wvmodel:
                vec += self.wvmodel[token]
        vec /= np.linalg.norm(vec)
        return vec

    def encode(self, shorttext):
        """ Calculated the encoded representation of the given text.

        With the trained autoencoder model, calculated the encoded representation.

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised.

        :param shorttext: a short sentence
        :return: encoded representation of the given short sentence
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        embedvec = self.shorttext_to_embedvec(shorttext)
        return self.encoder.predict(np.array([embedvec]))

    def score(self, shorttext):
        """ Calculate the scores for all the class labels for the given short sentence.

        Given a short sentence, calculate the classification scores for all class labels,
        returned as a dictionary with key being the class labels, and values being the scores.
        If the short sentence is empty, or if other numerical errors occur, the score will be `numpy.nan`.
        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: a short sentence
        :return: a dictionary with keys being the class labels, and values being the corresponding classification scores
        :type shorttext: str
        :rtype: dict
        :raise: ModelNotTrainedException
        """
        encoded_vec = self.encode(shorttext)
        scoredict = {}
        for classtype in self.train_encoded_vecs:
            try:
                scoredict[classtype] = 1 - cosine(encoded_vec, self.train_encoded_vecs[classtype])
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict