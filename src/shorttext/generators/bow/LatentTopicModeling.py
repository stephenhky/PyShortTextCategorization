
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from deprecation import deprecated

from ...utils import textpreprocessing as textpreprocess, gensim_corpora as gc, classification_exceptions as e
from ...utils.textpreprocessing import tokenize


# abstract class
class LatentTopicModeler(ABC):
    """
    Abstract class for various topic modeler.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
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

        self.normalize = normalize
        self.trained = False

    @deprecated(deprecated_in="4.0.0", removed_in="5.0.0")
    def generate_corpus(self, classdict: dict[str, list[str]]) -> None:
        """ Calculate the gensim dictionary and corpus, and extract the class labels
        from the training data. Called by :func:`~train`.

        :param classdict: training data
        :return: None
        :type classdict: dict
        """
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda sent: tokenize(self.preprocess_func(sent)))
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
        self.nb_topics = nb_topics
        raise e.NotImplementedException()

    def retrieve_bow(self, shorttext: str) -> None:
        """ Calculate the gensim bag-of-words representation of the given short text.

        :param shorttext: text to be represented
        :return: corpus representation of the text
        :type shorttext: str
        :rtype: list
        """
        return self.dictionary.doc2bow(tokenize(self.preprocess_func(shorttext)))

    def retrieve_bow_vector(self, shorttext, normalize=True):
        """ Calculate the vector representation of the bag-of-words in terms of numpy.ndarray.

        :param shorttext: short text
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :return: vector represtation of the text
        :type shorttext: str
        :type normalize: bool
        :rtype: numpy.ndarray
        """
        bow = self.retrieve_bow(shorttext)
        vec = np.zeros(len(self.dictionary))
        for id, val in bow:
            vec[id] = val
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec

    @abstractmethod
    def retrieve_topicvec(self, shorttext):
        """ Calculate the topic vector representation of the short text.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        :raise: NotImplementedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        raise e.NotImplementedException()

    @abstractmethod
    def get_batch_cos_similarities(self, shorttext):
        """ Calculate the cosine similarities of the given short text and all the class labels.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        :raise: NotImplementedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        raise e.NotImplementedException()

    def __getitem__(self, shorttext):
        return self.retrieve_topicvec(shorttext)

    def __contains__(self, shorttext):
        if not self.trained:
            raise e.ModelNotTrainedException()
        return True

    @abstractmethod
    def loadmodel(self, nameprefix):
        """ Load the model from files.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        :raise: NotImplementedException
        :type nameprefix: str
        """
        raise e.NotImplementedException()

    @abstractmethod
    def savemodel(self, nameprefix):
        """ Save the model to files.

        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        :raise: NotImplementedException
        :type nameprefix: str
        """
        raise e.NotImplementedException()