
from typing import Optional, Annotated

import numpy as np
import numpy.typing as npt
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.models import Model

from ....utils import kerasmodel_io as kerasio
from ....utils.classification_exceptions import ModelNotTrainedException
from ....utils.textpreprocessing import tokenize
from ....utils.compactmodel_io import CompactIOMachine
from ...base import AbstractScorer


class VarNNSumEmbeddedVecClassifier(AbstractScorer, CompactIOMachine):
    """Neural network classifier using summed embeddings.

    Wraps Keras neural network models for supervised short text classification.
    Each token is converted to an embedded vector using a pre-trained word-embedding
    model. The sentence embedding is the sum of token embeddings, normalized to
    a unit vector.

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
            maxlen: int = 15
    ):
        """Initialize the classifier.

        Args:
            wvmodel: Word embedding model (e.g., Word2Vec).
            vecsize: Vector size. Default: None (extracted from model).
            maxlen: Maximum number of words per sentence. Default: 15.
        """
        CompactIOMachine.__init__(
            self,
            {'classifier': 'sumnnlibvec'},
            'sumnnlibvec',
            ['_classlabels.txt', '.json', '.weights.h5']
        )
        self.wvmodel = wvmodel
        self.vecsize = self.wvmodel.vector_size if vecsize is None else vecsize
        self.maxlen = maxlen
        self.trained = False

    def convert_traindata_embedvecs(
            self,
            classdict: dict[str, list[str]]
    ) -> tuple[list[str], Annotated[npt.NDArray[np.float64], "2D Array"], Annotated[npt.NDArray[np.int64], "2D Array"]]:
        """Convert training data to embedded vectors.

        Converts each short text into a normalized sum of word embeddings.

        Args:
            classdict: Training data with class labels as keys and texts as values.

        Returns:
            Tuple of (class_labels, embedding_matrix, labels_array).
        """
        classlabels = sorted(classdict.keys())
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        indices = []
        embedvecs = []
        for classlabel in classlabels:
            for shorttext in classdict[classlabel]:
                embedvec = np.sum(
                    np.array([
                        self.word_to_embedvec(token)
                        for token in tokenize(shorttext)
                    ]),
                    axis=0
                )
                norm = np.linalg.norm(embedvec)
                if norm == 0:
                    continue
                embedvec /= norm
                embedvecs.append(embedvec)
                category_bucket = [0]*len(classlabels)
                category_bucket[lblidx_dict[classlabel]] = 1
                indices.append(category_bucket)

        indices = np.array(indices)
        embedvecs = np.array(embedvecs)
        return classlabels, embedvecs, indices

    def train(
            self,
            classdict: dict[str, list[str]],
            kerasmodel: Model,
            nb_epoch: int = 10
    ) -> None:
        """Train the classifier.

        Args:
            classdict: Training data.
            kerasmodel: Keras Sequential model.
            nb_epoch: Number of training epochs. Default: 10.

        Raises:
            ModelNotTrainedException: If not trained or loaded.
        """
        self.classlabels, train_embedvec, indices = self.convert_traindata_embedvecs(classdict)
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

    def loadmodel(self, nameprefix: str) -> None:
        """Load a trained model from files.

        Args:
            nameprefix: Prefix for input files.
        """
        self.model = kerasio.load_model(nameprefix)
        self.classlabels = [s.strip() for s in open(nameprefix+'_classlabels.txt', 'r')]
        self.trained = True

    def word_to_embedvec(self, word: str) -> Annotated[npt.NDArray[np.float64], "1D Array"]:
        """Convert a word to its embedding vector.

        Args:
            word: Input word.

        Returns:
            Embedding vector. Returns zeros if word not in vocabulary.
        """
        return self.wvmodel[word].astype(np.float64) if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_embedvec(self, shorttext: str) -> Annotated[npt.NDArray[np.float64], "1D Array"]:
        """Convert short text to embedding vector.

        Sums token embeddings and normalizes to unit vector.

        Args:
            shorttext: Input text.

        Returns:
            Normalized embedding vector.
        """
        vec = np.sum([
            self.wvmodel[token].astype(np.float64)
            for token in tokenize(shorttext)
            if token in self.wvmodel
        ])
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec /= np.linalg.norm(vec)
        return vec

    def score(self, shorttext: str) -> dict[str, float]:
        """Calculate classification scores for all class labels.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()

        embedvec = np.array(self.shorttext_to_embedvec(shorttext))
        predictions = self.model.predict(np.array([embedvec]))

        scoredict = {
            classlabel: predictions[0, idx]
            for idx, classlabel in enumerate(self.classlabels)
        }
        return scoredict


def load_varnnsumvec_classifier(
        wvmodel: KeyedVectors,
        name: str,
        compact: bool = True,
        vecsize: Optional[int] = None
) -> VarNNSumEmbeddedVecClassifier:
    """Load a VarNNSumEmbeddedVecClassifier from file.

    Args:
        wvmodel: Word embedding model.
        name: Model name (compact) or file prefix (non-compact).
        compact: Whether to load compact model. Default: True.
        vecsize: Vector size. Default: None.

    Returns:
        VarNNSumEmbeddedVecClassifier instance.
    """
    classifier = VarNNSumEmbeddedVecClassifier(wvmodel, vecsize=vecsize)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier
