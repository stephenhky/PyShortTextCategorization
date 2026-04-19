
import json
import pickle
from typing import Optional, Any
from collections import Counter

import numpy as np
import numpy.typing as npt
import sparse
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import orjson

from .LatentTopicModeling import LatentTopicModeler
from ...utils import kerasmodel_io as kerasio, textpreprocessing as textpreprocess
from ...utils.compactmodel_io import CompactIOMachine
from ...utils.classification_exceptions import ModelNotTrainedException
from ...utils.dtm import generate_npdict_document_term_matrix, convert_classdict_to_corpus
from ...utils.compute import cosine_similarity
from ...schemas.models import AutoEncoderPackage


autoencoder_suffices = ['_encoder.json', '_encoder.weights.h5', '_classtopicvecs.pkl',
                        '_decoder.json', '_decoder.weights.h5', '_autoencoder.json', '_autoencoder.weights.h5',
                        '.json']


def get_autoencoder_models(
        vector_size: int,
        nb_latent_vector_size: int
) -> AutoEncoderPackage:
    """Create autoencoder model components.

    Args:
        vector_size: Size of input vectors.
        nb_latent_vector_size: Size of the latent space (number of topics).

    Returns:
        AutoEncoderPackage containing autoencoder, encoder, and decoder models.
    """
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
    """Topic modeler using autoencoder.

    Uses a Keras autoencoder to learn latent topic representations.
    The encoded vectors serve as topic vectors for short text classification.

    Reference:
        Francois Chollet, "Building Autoencoders in Keras,"
        https://blog.keras.io/building-autoencoders-in-keras.html
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
        """Train the autoencoder topic model.

        Args:
            classdict: Training data with class labels as keys and texts as values.
            nb_topics: Number of latent topics (encoding dimensions).
            *args: Arguments for Keras model fitting.
            **kwargs: Keyword arguments for Keras model fitting.
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
        """Get bag-of-words representation.

        Args:
            shorttext: Input text.

        Returns:
            List of (token_index, count) tuples.
        """
        tokens_freq = Counter(self.tokenize_func(self.preprocess_func(shorttext)))
        return [
            (self.token2indices[token], freq)
            for token, freq in tokens_freq.items()
            if token in self.token2indices.keys()
        ]

    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get bag-of-words vector.

        Args:
            shorttext: Input text.

        Returns:
            BOW vector (normalized if normalize=True).
        """
        bow = self.retrieve_bow(shorttext)
        if len(bow) > 0:
            vec = sparse.COO(
                [[0]*len(bow), [id for id, val in bow]],
                [val for id, val in bow],
                shape=(1, len(self.token2indices))
            ).todense()[0]
        else:
            vec = np.ones(len(self.token2indices))
        if self.normalize:
            vec = vec.astype(np.float64) / np.linalg.norm(vec)
        return vec

    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get topic vector for short text.

        Args:
            shorttext: Input text.

        Returns:
            Encoded vector representation.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        bow_vector = self.retrieve_bow_vector(shorttext)
        encoded_vec = self.encoder.predict(np.expand_dims(bow_vector, axis=0))[0]
        if self.normalize:
            encoded_vec /= np.linalg.norm(encoded_vec)
        return encoded_vec.astype(np.float64)

    def precalculate_liststr_topicvec(self, shorttexts: list[str]) -> npt.NDArray[np.float64]:
        """Calculate average topic vector for a list of texts.

        Used during training to compute class centroids.

        Args:
            shorttexts: List of texts.

        Returns:
            Average topic vector (normalized).

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        sumvec = sum([self.retrieve_topicvec(shorttext) for shorttext in shorttexts])
        sumvec /= np.linalg.norm(sumvec)
        return sumvec

    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
        """Get cosine similarities to all class centroids.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to similarity scores.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        simdict = {}
        for label, classtopicvec in self.classtopicvecs.items():
            simdict[label] = cosine_similarity(
                classtopicvec, self.retrieve_topicvec(shorttext)
            )
        return simdict

    def savemodel(self, nameprefix: str, save_complete_autoencoder: bool=True) -> None:
        """Save the autoencoder model to files.

        Saves encoder, optional decoder, and autoencoder weights along with
        configuration parameters.

        Args:
            nameprefix: Prefix for output files.
            save_complete_autoencoder: Whether to save decoder and complete autoencoder. Default: True.

        Raises:
            ModelNotTrainedException: If model not trained.
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
        """Load the autoencoder model from files.

        Args:
            nameprefix: Prefix for input files.
            load_incomplete: If True, only load encoder (for models from v0.2.1). Default: False.

        Raises:
            ModelNotTrainedException: If loading fails.
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

    def get_info(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with model information.
        """
        return CompactIOMachine.get_info(self)


def load_autoencoder_topicmodel(
        name: str,
        preprocessor: Optional[callable] = None,
        tokenizer: Optional[callable] = None,
        compact: bool=True
) -> AutoencodingTopicModeler:
    """Load an autoencoder topic model from files.

    Args:
        name: Model name (compact) or file prefix (non-compact).
        preprocessor: Text preprocessing function.
        compact: Whether to load compact model. Default: True.

    Returns:
        An AutoencodingTopicModeler instance.
    """
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()

    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor, tokenizer=tokenizer)
    if compact:
        autoencoder.load_compact_model(name)
    else:
        autoencoder.loadmodel(name)
    return autoencoder
