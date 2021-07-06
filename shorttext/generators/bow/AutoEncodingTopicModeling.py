
import json
import pickle
from functools import reduce
from operator import add

import numpy as np
from gensim.corpora import Dictionary
from keras import Input
from keras import Model
from keras.layers import Dense
from scipy.spatial.distance import cosine

from .LatentTopicModeling import LatentTopicModeler
from shorttext.utils import kerasmodel_io as kerasio, textpreprocessing as textpreprocess
from shorttext.utils.compactmodel_io import CompactIOMachine
from shorttext.utils.classification_exceptions import ModelNotTrainedException


autoencoder_suffices = ['.gensimdict', '_encoder.json', '_encoder.h5', '_classtopicvecs.pkl',
                        '_decoder.json', '_decoder.h5', '_autoencoder.json', '_autoencoder.h5',
                        '.json']


class AutoencodingTopicModeler(LatentTopicModeler, CompactIOMachine):
    """
    This class facilitates the topic modeling of input training data using the autoencoder.

    A reference about how an autoencoder is written with keras by Francois Chollet, titled
    `Building Autoencoders in Keras
    <https://blog.keras.io/building-autoencoders-in-keras.html>`_ .

    This class extends :class:`LatentTopicModeler`.
    """
    def train(self, classdict, nb_topics, *args, **kwargs):
        """ Train the autoencoder.

        :param classdict: training data
        :param nb_topics: number of topics, i.e., the number of encoding dimensions
        :param args: arguments to be passed to keras model fitting
        :param kwargs: arguments to be passed to keras model fitting
        :return: None
        :type classdict: dict
        :type nb_topics: int
        """
        CompactIOMachine.__init__(self, {'classifier': 'kerasautoencoder'}, 'kerasautoencoder', autoencoder_suffices)
        self.nb_topics = nb_topics
        self.generate_corpus(classdict)
        vecsize = len(self.dictionary)

        # define all the layers of the autoencoder
        input_vec = Input(shape=(vecsize,))
        encoded = Dense(self.nb_topics, activation='relu')(input_vec)
        decoded = Dense(vecsize, activation='sigmoid')(encoded)

        # define the autoencoder model
        autoencoder = Model(input=input_vec, output=decoded)

        # define the encoder
        encoder = Model(input=input_vec, output=encoded)

        # define the decoder
        encoded_input = Input(shape=(self.nb_topics,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        # compile the autoencoder
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # process training data
        embedvecs = np.array(reduce(add,
                                    [[self.retrieve_bow_vector(shorttext, normalize=True) for shorttext in classdict[classtype]]
                                     for classtype in classdict]
                                    )
                             )

        # fit the model
        autoencoder.fit(embedvecs, embedvecs, *args, **kwargs)

        # store the autoencoder models
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        # flag setting
        self.trained = True

        # classes topic vector precomputation
        self.classtopicvecs = {}
        for label in classdict:
            self.classtopicvecs[label] = self.precalculate_liststr_topicvec(classdict[label])

    def retrieve_topicvec(self, shorttext):
        """ Calculate the topic vector representation of the short text.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: short text
        :return: encoded vector representation of the short text
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        if not self.trained:
            raise ModelNotTrainedException()
        bow_vector = self.retrieve_bow_vector(shorttext)
        encoded_vec = self.encoder.predict(np.array([bow_vector]))[0]
        if self.normalize:
            encoded_vec /= np.linalg.norm(encoded_vec)
        return encoded_vec

    def precalculate_liststr_topicvec(self, shorttexts):
        """ Calculate the summed topic vectors for training data for each class.

        This function is called while training.

        :param shorttexts: list of short texts
        :return: average topic vector
        :raise: ModelNotTrainedException
        :type shorttexts: list
        :rtype: numpy.ndarray
        """
        sumvec = sum([self.retrieve_topicvec(shorttext) for shorttext in shorttexts])
        sumvec /= np.linalg.norm(sumvec)
        return sumvec

    def get_batch_cos_similarities(self, shorttext):
        """ Calculate the score, which is the cosine similarity with the topic vector of the model,
        of the short text against each class labels.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: short text
        :return: dictionary of scores of the text to all classes
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: dict
        """
        if not self.trained:
            raise ModelNotTrainedException()
        simdict = {}
        for label in self.classtopicvecs:
            simdict[label] = 1 - cosine(self.classtopicvecs[label], self.retrieve_topicvec(shorttext))
        return simdict

    def savemodel(self, nameprefix, save_complete_autoencoder=True):
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There are files with names ending with "_encoder.json" and "_encoder.h5", which are
        the JSON and HDF5 files for the encoder respectively. They also include a gensim dictionary (.gensimdict).

        If `save_complete_autoencoder` is True,
        then there are also files with names ending with "_decoder.json" and "_decoder.h5".

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param nameprefix: prefix of the paths of the file
        :param save_complete_autoencoder: whether to store the decoder and the complete autoencoder (Default: True; but False for version <= 0.2.1)
        :return: None
        :type nameprefix: str
        :type save_complete_autoencoder: bool
        """
        if not self.trained:
            raise ModelNotTrainedException()

        parameters = {}
        parameters['nb_topics'] = self.nb_topics
        parameters['classlabels'] = self.classlabels
        json.dump(parameters, open(nameprefix+'.json', 'wb'))

        self.dictionary.save(nameprefix+'.gensimdict')
        kerasio.save_model(nameprefix+'_encoder', self.encoder)
        if save_complete_autoencoder:
            kerasio.save_model(nameprefix+'_decoder', self.decoder)
            kerasio.save_model(nameprefix+'_autoencoder', self.autoencoder)
        pickle.dump(self.classtopicvecs, open(nameprefix+'_classtopicvecs.pkl', 'wb'))

    def loadmodel(self, nameprefix, load_incomplete=False):
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, load the model into files, with name given by the prefix.
        There are files with names ending with "_encoder.json" and "_encoder.h5", which are
        the JSON and HDF5 files for the encoder respectively.
        They also include a gensim dictionary (.gensimdict).

        :param nameprefix: prefix of the paths of the file
        :param load_incomplete: load encoder only, not decoder and autoencoder file (Default: False; put True for model built in version <= 0.2.1)
        :return: None
        :type nameprefix: str
        :type load_incomplete: bool
        """
        # load the JSON file (parameters)
        parameters = json.load(open(nameprefix+'.json', 'r'))
        self.nb_topics = parameters['nb_topics']
        self.classlabels = parameters['classlabels']

        self.dictionary = Dictionary.load(nameprefix + '.gensimdict')
        self.encoder = kerasio.load_model(nameprefix+'_encoder')
        self.classtopicvecs = pickle.load(open(nameprefix+'_classtopicvecs.pkl', 'rb'))
        if not load_incomplete:
            self.decoder = kerasio.load_model(nameprefix+'_decoder')
            self.autoencoder = kerasio.load_model(nameprefix+'_autoencoder')
        self.trained = True


def load_autoencoder_topicmodel(name,
                                preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                compact=True):
    """ Load the autoencoding topic model from files.

    :param name: name (if compact=True) or prefix (if compact=False) of the paths of the model files
    :param preprocessor: function that preprocesses the text. (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: an autoencoder as a topic modeler
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: generators.bow.AutoEncodingTopicModeling.AutoencodingTopicModeler
    """
    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor)
    if compact:
        autoencoder.load_compact_model(name)
    else:
        autoencoder.loadmodel(name)
    return autoencoder
