
import json
import pickle
from functools import reduce
from operator import add
from typing import Optional, Any
from collections import Counter

import numpy as np
import numpy.typing as npt
import sparse
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from scipy.spatial.distance import cosine
import orjson

from .LatentTopicModeling import LatentTopicModeler
from ...utils import kerasmodel_io as kerasio, textpreprocessing as textpreprocess
from ...utils.compactmodel_io import CompactIOMachine
from ...utils.classification_exceptions import ModelNotTrainedException
from ...utils.dtm import generate_npdict_document_term_matrix, convert_classdict_to_corpus
from ...schemas.models import AutoEncoderPackage


autoencoder_suffices = ['_encoder.json', '_encoder.weights.h5', '_classtopicvecs.pkl',
                        '_decoder.json', '_decoder.weights.h5', '_autoencoder.json', '_autoencoder.weights.h5',
                        '.json']


def get_autoencoder_models(
        vector_size: int,
        nb_latent_vector_size: int
) -> AutoEncoderPackage:
    # define all the layers of the autoencoder
    input_vec = Input(shape=(vector_size,))
    encoded = Dense(nb_latent_vector_size, activation='relu')(input_vec)
    decoded = Dense(vector_size, activation='sigmoid')(encoded)

    # define the autoencoder model
    autoencoder = Model(inputs=input_vec, outputs=decoded)

    # define the encoder
    encoder = Model(inputs=input_vec, outputs=encoded)

    # define the decoder
    encoded_input = Input(shape=(nb_latent_vector_size,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    # compile the autoencoder
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return AutoEncoderPackage(
        autoencoder=autoencoder,
        encoder=encoder,
        decoder=decoder
    )


class AutoencodingTopicModeler(LatentTopicModeler, CompactIOMachine):
    """
    This class facilitates the topic modeling of input training data using the autoencoder.

    A reference about how an autoencoder is written with keras by Francois Chollet, titled
    `Building Autoencoders in Keras
    <https://blog.keras.io/building-autoencoders-in-keras.html>`_ .

    This class extends :class:`LatentTopicModeler`.
    """

    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            normalize: bool = True
    ):
        CompactIOMachine.__init__(self, {'classifier': 'kerasautoencoder'}, 'kerasautoencoder', autoencoder_suffices)
        LatentTopicModeler.__init__(self, preprocessor, tokenizer, normalize=normalize)

    def train(self, classdict: dict[str, list[str]], nb_topics: int, *args, **kwargs) -> None:
        """ Train the autoencoder.

        :param classdict: training data
        :param nb_topics: number of topics, i.e., the number of encoding dimensions
        :param args: arguments to be passed to keras model fitting
        :param kwargs: arguments to be passed to keras model fitting
        :return: None
        :type classdict: dict
        :type nb_topics: int
        """
        self.nb_topics = nb_topics
        corpus, docids = convert_classdict_to_corpus(classdict, self.preprocess_func)
        dtm_matrix = generate_npdict_document_term_matrix(
            corpus, docids, tokenize_func=self.tokenize_func
        )
        vecsize = dtm_matrix.dimension_sizes[1]
        self.token2indices = dtm_matrix._keystrings_to_indices[1]
        self.classlabels = sorted(classdict.keys())

        autoencoder_package = get_autoencoder_models(vecsize, self.nb_topics)
        autoencoder = autoencoder_package.autoencoder
        encoder = autoencoder_package.encoder
        decoder = autoencoder_package.decoder

        # process training data
        # embedvecs = np.array(
        #     reduce(
        #         add,
        #         [
        #             [
        #                 self.retrieve_bow_vector(shorttext)
        #                 for shorttext in classdict[classtype]
        #             ]
        #             for classtype in classdict
        #         ]
        #     )
        # )
        embedvecs = dtm_matrix.to_numpy()

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

    def retrieve_bow(self, shorttext: str) -> list[tuple[int, int]]:
        tokens_freq = Counter(self.tokenize_func(self.preprocess_func(shorttext)))
        return [
            (self.token2indices[token], freq)
            for token, freq in tokens_freq.items()
            if token in self.token2indices.keys()
        ]

    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        bow = self.retrieve_bow(shorttext)
        if len(bow) > 0:
            vec = sparse.COO(
                [[0]*len(bow), [id for id, val in bow]],
                [val for id, val in bow],
                shape=(1, len(self.token2indices))
            ).todense()[0]
        else:
            vec = np.repeat(1., len(self.token2indices))
        if self.normalize:
            vec = np.array(vec, dtype=np.float64) / np.linalg.norm(vec)
        return vec

    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
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
        encoded_vec = self.encoder.predict(np.expand_dims(bow_vector, axis=0))[0]
        if self.normalize:
            encoded_vec /= np.linalg.norm(encoded_vec)
        return encoded_vec

    def precalculate_liststr_topicvec(self, shorttexts: list[str]) -> npt.NDArray[np.float64]:
        """ Calculate the summed topic vectors for training data for each class.

        This function is called while training.

        :param shorttexts: list of short texts
        :return: average topic vector
        :raise: ModelNotTrainedException
        :type shorttexts: list
        :rtype: numpy.ndarray
        """
        sumvec = sum([self.retrieve_topicvec(shorttext) for shorttext in shorttexts])   # correct, but should be refined
        sumvec /= np.linalg.norm(sumvec)
        return sumvec

    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
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

    def savemodel(self, nameprefix: str, save_complete_autoencoder: bool=True) -> None:
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There are files with names ending with "_encoder.json" and "_encoder.weights.h5", which are
        the JSON and HDF5 files for the encoder respectively. They also include a gensim dictionary (.gensimdict).

        If `save_complete_autoencoder` is True,
        then there are also files with names ending with "_decoder.json" and "_decoder.weights.h5".

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
        parameters['tokens2indices'] = self.token2indices
        open(nameprefix + '.json', 'wb').write(orjson.dumps(parameters))
        kerasio.save_model(nameprefix+'_encoder', self.encoder)
        if save_complete_autoencoder:
            kerasio.save_model(nameprefix+'_decoder', self.decoder)
            kerasio.save_model(nameprefix+'_autoencoder', self.autoencoder)
        pickle.dump(self.classtopicvecs, open(nameprefix+'_classtopicvecs.pkl', 'wb'))

    def loadmodel(self, nameprefix: str, load_incomplete: bool=False) -> None:
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, load the model into files, with name given by the prefix.
        There are files with names ending with "_encoder.json" and "_encoder.weights.h5", which are
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
        self.token2indices = parameters['tokens2indices']
        self.encoder = kerasio.load_model(nameprefix+'_encoder')
        self.classtopicvecs = pickle.load(open(nameprefix+'_classtopicvecs.pkl', 'rb'))
        if not load_incomplete:
            self.decoder = kerasio.load_model(nameprefix+'_decoder')
            self.autoencoder = kerasio.load_model(nameprefix+'_autoencoder')
        self.trained = True

    def getinfo(self) -> dict[str, Any]:
        return super(CompactIOMachine).getinfo()


def load_autoencoder_topicmodel(
        name: str,
        preprocessor: Optional[callable] = None,
        tokenizer: Optional[callable] = None,
        compact: bool=True
) -> AutoencodingTopicModeler:
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
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()

    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor, tokenizer=tokenizer)
    if compact:
        autoencoder.load_compact_model(name)
    else:
        autoencoder.loadmodel(name)
    return autoencoder
