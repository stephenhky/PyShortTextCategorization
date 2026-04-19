
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import numpy.typing as npt

from ...utils import textpreprocessing as textpreprocess, classification_exceptions as e
from ...utils.textpreprocessing import tokenize


# abstract class
class LatentTopicModeler(ABC):
    """Abstract base class for topic modelers.

    Provides interface for converting short texts to topic vector
    representations using various topic modeling algorithms.
    """

    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            normalize: bool = True
    ):
        """Initialize the topic modeler.

        Args:
            preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
            tokenizer: Tokenization function. Default: tokenize.
            normalize: Whether to normalize output vectors. Default: True.
        """
        if preprocessor is None:
            self.preprocess_func = textpreprocess.standard_text_preprocessor_1()
        else:
            self.preprocess_func = preprocessor
        if tokenizer is None:
            self.tokenize_func = tokenize
        else:
            self.tokenize_func = tokenizer

        self.normalize = normalize
        self.trained = False

    @abstractmethod
    def train(self, classdict: dict[str, list[str]], nb_topics: int, *args, **kwargs) -> None:
        """Train the topic modeler.

        Args:
            classdict: Training data with class labels as keys and texts as values.
            nb_topics: Number of latent topics.
            *args: Additional arguments for the training algorithm.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def retrieve_bow(self, shorttext: str) -> list[tuple[int, int]]:
        """Get bag-of-words representation.

        Args:
            shorttext: Input text.

        Returns:
            List of (word_id, count) tuples.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get bag-of-words vector.

        Args:
            shorttext: Input text.

        Returns:
            BOW vector.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get topic vector for short text.

        Args:
            shorttext: Input text.

        Returns:
            Topic vector.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
        """Get cosine similarities to all classes.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to similarity scores.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    def __getitem__(self, shorttext) -> npt.NDArray[np.float64]:
        """Get topic vector for text (shortcut for retrieve_topicvec)."""
        return self.retrieve_topicvec(shorttext)

    def __contains__(self, shorttext):
        """Check if model is trained."""
        if not self.trained:
            raise e.ModelNotTrainedException()
        return True

    @abstractmethod
    def loadmodel(self, nameprefix: str):
        """Load model from files.

        Args:
            nameprefix: Prefix for input files.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def savemodel(self, nameprefix: str):
        """Save model to files.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with model information.
        """
        raise NotImplemented()
