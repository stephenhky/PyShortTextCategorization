
import pickle
from collections import defaultdict
from typing import Optional, Annotated

import numpy as np
import numpy.typing as npt
from gensim.models.keyedvectors import KeyedVectors
from loguru import logger

from ....utils.classification_exceptions import ModelNotTrainedException
from ....utils import shorttext_to_avgvec
from ....utils.compactmodel_io import CompactIOMachine
from ....utils.compute import cosine_similarity


class SumEmbeddedVecClassifier(CompactIOMachine):
    """Classifier using summed word embeddings.

    Each class is represented as the sum of word embeddings for its
    training sentences, normalized to a unit vector. Prediction uses
    cosine similarity between the input vector and class centroids.

    Reference:
        Pre-trained Word2Vec: https://code.google.com/archive/p/word2vec/
    """

    def __init__(
            self,
            wvmodel: KeyedVectors,
            vecsize: Optional[int] = None,
            simfcn: Optional[callable] = None
    ):
        """Initialize the classifier.

        Args:
            wvmodel: Word embedding model (e.g., Word2Vec).
            vecsize: Vector size. Default: None (extracted from model).
            simfcn: Similarity function. Default: cosine_similarity.
        """
        CompactIOMachine.__init__(
            self,
            {'classifier': 'sumvec'},
            'sumvec',
            ['_embedvecdict.pkl']
        )
        self.wvmodel = wvmodel
        self.vecsize = self.wvmodel.vector_size if vecsize is None else vecsize
        self.simfcn = simfcn if simfcn is not None else cosine_similarity
        self.trained = False

    def train(self, classdict: dict[str, list[str]]) -> None:
        """Train the classifier.

        Args:
            classdict: Training data with class labels as keys and texts as values.

        Raises:
            ModelNotTrainedException: If not trained or loaded.
        """
        self.addvec = defaultdict(lambda : np.zeros(self.vecsize))
        for classtype in classdict:
            self.addvec[classtype] = np.sum(
                [
                    self.shorttext_to_embedvec(shorttext)
                    for shorttext in classdict[classtype]
                ],
                axis=0
            )
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.trained = True

    def savemodel(self, nameprefix: str) -> None:
        """Save the trained model.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        pickle.dump(self.addvec, open(nameprefix+'_embedvecdict.pkl', 'wb'))

    def loadmodel(self, nameprefix: str) -> None:
        """Load a trained model.

        Args:
            nameprefix: Prefix for input files.
        """
        self.addvec = pickle.load(open(nameprefix+'_embedvecdict.pkl', 'rb'))
        self.trained = True

    def shorttext_to_embedvec(
            self,
            shorttext: str
    ) -> Annotated[npt.NDArray[np.float64], "1D Array"]:
        """Convert short text to embedding vector.

        Args:
            shorttext: Input text.

        Returns:
            Normalized embedding vector.
        """
        return shorttext_to_avgvec(shorttext, self.wvmodel)

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

        vec = self.shorttext_to_embedvec(shorttext)
        scoredict = {}
        for classtype, addvec in self.addvec.items():
            try:
                scoredict[classtype] = self.simfcn(vec, addvec)
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict


def load_sumword2vec_classifier(
        wvmodel: KeyedVectors,
        name: str,
        compact: bool = True,
        vecsize: Optional[int] = None
) -> SumEmbeddedVecClassifier:
    """Load a SumEmbeddedVecClassifier from file.

    Args:
        wvmodel: Word embedding model.
        name: Model name (compact) or prefix (non-compact).
        compact: Whether to load compact model. Default: True.
        vecsize: Vector size. Default: None.

    Returns:
        SumEmbeddedVecClassifier instance.
    """
    classifier = SumEmbeddedVecClassifier(wvmodel, vecsize=vecsize)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier
