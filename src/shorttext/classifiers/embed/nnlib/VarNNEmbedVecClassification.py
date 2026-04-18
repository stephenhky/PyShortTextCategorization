
import os
import warnings
from typing import Any, Optional, Annotated

import numpy as np
import numpy.typing as npt
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.models import Model
import orjson

from ....utils import kerasmodel_io as kerasio
from ....utils.classification_exceptions import ModelNotTrainedException
from ....utils import tokenize
from ....utils.compactmodel_io import CompactIOMachine
from ...base import AbstractScorer


class VarNNEmbeddedVecClassifier(AbstractScorer, CompactIOMachine):
    """Neural network classifier for short text categorization.

    Wraps Keras neural network models for supervised short text classification.
    Each token is converted to an embedded vector using a pre-trained word-embedding
    model (e.g., Word2Vec). Sentences are represented as matrices (rank-2 or rank-3 arrays)
    and processed by the neural network.

    The neural network model must be a Keras Sequential model with output dimension
    matching the number of class labels.

    Reference:
        Pre-trained Word2Vec: https://code.google.com/archive/p/word2vec/
        Example models available in the frameworks module.
    """

    def __init__(
            self,
            wvmodel: KeyedVectors,
            vecsize: Optional[int] = None,
            maxlen: int = 15,
            with_gensim: bool = False
    ):
        """Initialize the classifier.

        Args:
            wvmodel: Word embedding model (e.g., Word2Vec).
            vecsize: Vector size. Default: None (extracted from model).
            maxlen: Maximum number of words per sentence. Default: 15.
            with_gensim: Whether to use gensim format. Default: False.
        """
        CompactIOMachine.__init__(
            self,
            {'classifier': 'nnlibvec'},
            'nnlibvec',
            ['_classlabels.txt', '.json', '.weights.h5', '_config.json']
        )
        self.wvmodel = wvmodel
        self.vecsize = self.wvmodel.vector_size if vecsize is None else vecsize
        self.maxlen = maxlen
        self.with_gensim = False if not with_gensim else with_gensim
        self.trained = False

    def convert_trainingdata_matrix(
            self,
            classdict: dict[str, list[str]]
    ) -> tuple[list[str], Annotated[npt.NDArray[np.float64], "3D Array"], Annotated[npt.NDArray[np.int64], "2D Array"]]:
        """Convert training data to neural network input format.

        Args:
            classdict: Training data with class labels as keys and texts as values.

        Returns:
            Tuple of (class_labels, embedded_vectors, labels_array).
        """
        classlabels = sorted(classdict.keys())   # sort the class labels to ensure uniqueness
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        phrases = []
        indices = []
        for label in classlabels:
            for shorttext in classdict[label]:
                shorttext = shorttext if isinstance(shorttext, str) else ''
                category_bucket = [0]*len(classlabels)
                category_bucket[lblidx_dict[label]] = 1
                indices.append(category_bucket)
                phrases.append(tokenize(shorttext))

        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j, :] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int_)

        return classlabels, train_embedvec, indices

    def train(
            self,
            classdict: dict[str, list[str]],
            kerasmodel: Model,
            nb_epoch: int = 10
    ):
        """Train the classifier.

        Args:
            classdict: Training data.
            kerasmodel: Keras Sequential model.
            nb_epoch: Number of training epochs. Default: 10.

        Raises:
            ModelNotTrainedException: If model not loaded.
        """
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix(classdict)
        kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)
        self.model = kerasmodel
        self.trained = True

    def savemodel(self, nameprefix: str) -> None:
        """Save the trained model to files.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()

        kerasio.save_model(nameprefix, self.model)
        open(nameprefix+'_classlabels.txt', 'w').write('\n'.join(self.classlabels))
        open(nameprefix + '_config.json', 'wb').write(
            orjson.dumps(
                {'with_gensim': False, 'maxlen': self.maxlen, 'vecsize': self.vecsize}
            )
        )

    def loadmodel(self, nameprefix: str) -> None:
        """Load a trained model from files.

        Args:
            nameprefix: Prefix for input files.
        """
        self.model = kerasio.load_model(nameprefix)
        self.classlabels = [line.strip() for line in open(nameprefix+'_classlabels.txt', 'r')]

        if os.path.exists(nameprefix+'_config.json'):
            config = orjson.loads(open(nameprefix+'_config.json', 'rb').read())
            if 'maxlen' in config:
                self.maxlen = config['maxlen']
            else:
                self.maxlen = 15
            if 'vecsize' in config:
                self.vecsize = config['vecsize']
            else:
                self.vecsize = self.wvmodel.vector_size
            if self.vecsize != self.wvmodel.vector_size:
                warnings.warn(
                    f'Record vector size ({self.vecsize}) is not the same as that of the given word-embedding model ({self.wvmodel.vector_size})! ' + \
                    f'Setting the vector size to be {self.wvmodel.vector_size}, but there may be run time error!'
                )
                self.vecsize = self.wvmodel.vector_size
        else:
            self.maxlen = 15
            self.vecsize = self.wvmodel.vector_size
            warnings.warn('Model files from old versions.')

        self.with_gensim = False
        self.trained = True

    def word_to_embedvec(self, word: str) -> npt.NDArray[np.float64]:
        """Convert a word to its embedding vector.

        Args:
            word: Input word.

        Returns:
            Embedding vector. Returns zeros if word not in vocabulary.
        """
        return self.wvmodel[word].astype(np.float64) if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_matrix(
            self,
            shorttext: str
    ) -> Annotated[npt.NDArray[np.float64], "2D Array"]:
        """Convert short text to embedding matrix.

        Args:
            shorttext: Input text.

        Returns:
            Matrix of shape (maxlen, vecsize) with embedding vectors.
        """
        tokens = tokenize(shorttext)
        matrix = np.zeros((self.maxlen, self.vecsize))
        for i in range(min(self.maxlen, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix

    def score(
            self,
            shorttext: str,
            model_params: Optional[dict[str, Any]] = None
    ) -> dict[str, float]:
        """Calculate classification scores for all class labels.

        Args:
            shorttext: Input text.
            model_params: Additional parameters for model prediction.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if model_params is None:
            model_params = {}
        
        if not self.trained:
            raise ModelNotTrainedException()

        matrix = np.array([self.shorttext_to_matrix(shorttext)])
        predictions = self.model.predict(matrix, **model_params)

        score_dict = {
            classlabel: predictions[0, j]
            for j, classlabel in enumerate(self.classlabels)
        }

        return score_dict


def load_varnnlibvec_classifier(
        wvmodel: KeyedVectors,
        name: str,
        compact: bool = True,
        vecsize: Optional[int] = None
) -> VarNNEmbeddedVecClassifier:
    """Load a VarNNEmbeddedVecClassifier from file.

    Args:
        wvmodel: Word embedding model.
        name: Model name (compact) or file prefix (non-compact).
        compact: Whether to load compact model. Default: True.
        vecsize: Vector size. Default: None.

    Returns:
        VarNNEmbeddedVecClassifier instance.
    """
    classifier = VarNNEmbeddedVecClassifier(wvmodel, vecsize=vecsize)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier
