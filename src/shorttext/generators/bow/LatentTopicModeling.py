
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from ...utils import textpreprocessing as textpreprocess, classification_exceptions as e
from ...utils.textpreprocessing import tokenize
from ...utils.compactmodel_io import CompactIOMachine


# abstract class
class LatentTopicModeler(ABC, CompactIOMachine):
    """
    Abstract class for various topic modeler.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            normalize: bool = True
    ):
        """ Initialize the modeler.

        :param preprocessor: function that preprocesses the text. (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :type preprocessor: function
        :type normalize: bool
        """
        if preprocessor is None:
            self.preprocess_func = textpreprocess.standard_text_preprocessor_1()
        else:
            self.preprocess_func = preprocessor
        if tokenizer is None:
            self.tokenize_func = tokenize

        self.normalize = normalize
        self.trained = False

    @abstractmethod
    def train(self, classdict: dict[str, list[str]], nb_topics: int, *args, **kwargs) -> None:
        """ Train the modeler.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param classdict: training data
        :param nb_topics: number of latent topics
        :param args: arguments to be passed into the wrapped training functions
        :param kwargs: arguments to be passed into the wrapped training functions
        :return: None
        :raise: NotImplementedException
        :type classdict: dict
        :type nb_topics: int
        """
        raise NotImplemented()

    @abstractmethod
    def retrieve_bow(self, shorttext: str) -> list[tuple[int, int]]:
        raise NotImplemented()

    @abstractmethod
    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        raise NotImplemented()

    @abstractmethod
    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
        """ Calculate the topic vector representation of the short text.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        :raise: NotImplementedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        raise NotImplemented()

    @abstractmethod
    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
        """ Calculate the cosine similarities of the given short text and all the class labels.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        :raise: NotImplementedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        raise NotImplemented()

    def __getitem__(self, shorttext) -> npt.NDArray[np.float64]:
        return self.retrieve_topicvec(shorttext)

    def __contains__(self, shorttext):
        if not self.trained:
            raise e.ModelNotTrainedException()
        return True

    @abstractmethod
    def loadmodel(self, nameprefix: str):
        """ Load the model from files.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        :raise: NotImplementedException
        :type nameprefix: str
        """
        raise NotImplemented()

    @abstractmethod
    def savemodel(self, nameprefix: str):
        """ Save the model to files.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        :raise: NotImplementedException
        :type nameprefix: str
        """
        raise NotImplemented()