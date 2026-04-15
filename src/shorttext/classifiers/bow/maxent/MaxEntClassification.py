
from typing import Literal, Optional

import sparse
import orjson
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from ....utils import kerasmodel_io as kerasio
from ....utils import tokenize
from ....utils import classification_exceptions as e
from ....utils.compactmodel_io import CompactIOMachine
from ....utils.dtm import convert_classdict_to_xy
from ...base import AbstractScorer


def logistic_framework(
        nb_features: int,
        nb_outputs: int,
        l2reg: float = 0.01,
        bias_l2reg: float = 0.01,
        optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam"
) -> Model:
    """ Construct the neural network of maximum entropy classifier.

    Given the numbers of features and the output labels, return a keras neural network
     for implementing maximum entropy (multinomial) classifier.

    :param nb_features: number of features
    :param nb_outputs: number of output labels
    :param l2reg: L2 regularization coefficient (Default: 0.01)
    :param bias_l2reg: L2 regularization coefficient for bias (Default: 0.01)
    :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
    :return: keras sequential model for maximum entropy classifier
    :type nb_features: int
    :type nb_outputs: int
    :type l2reg: float
    :type bias_l2reg: float
    :type optimizer: str
    :rtype: keras.model.Sequential
    """
    kmodel = Sequential()
    kmodel.add(Dense(units=nb_outputs,
                     activation='softmax',
                     input_shape=(nb_features,),
                     kernel_regularizer=l2(l2reg),
                     bias_regularizer=l2(bias_l2reg))
               )
    kmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return kmodel


class MaxEntClassifier(AbstractScorer, CompactIOMachine):
    """
    This is a classifier that implements the principle of maximum entropy.

    Reference:
    * Adam L. Berger, Stephen A. Della Pietra, Vincent J. Della Pietra, "A Maximum Entropy Approach to Natural Language Processing," *Computational Linguistics* 22(1): 39-72 (1996).
    """

    def __init__(self, preprocessor: Optional[callable] = None):
        """ Initializer.

        :param preprocessor: text preprocessor
        :type preprocessor: function
        """
        CompactIOMachine.__init__(
            self,
            {'classifier': 'maxent'},
            'maxent',
            ['_classlabels.txt', '.json', '.weights.h5', '_labels2idx.json', '_tokens2idx.json']
        )

        if preprocessor is None:
            preprocessor = lambda s: s.lower()

        self.preprocess_func = preprocessor
        self.trained = False

    def shorttext_to_vec(self, shorttext: str) -> sparse.SparseArray:
        """ Convert the shorttext into a sparse vector given the dictionary.

        According to the dictionary (gensim.corpora.Dictionary), convert the given text
        into a vector representation, according to the occurence of tokens.

        This function is deprecated and no longer used because it is too slow to run in a loop.
        But this is used while doing prediction.

        :param shorttext: short text to be converted.
        :return: sparse vector of the vector representation
        :type shorttext: str
        :rtype: scipy.sparse.dok_matrix
        """
        tokens = tokenize(self.preprocess_func(shorttext))
        token_indices = [
            self.token2idx.get(token)
            for token in tokens
            if token in self.token2idx.keys()
        ]

        vec = sparse.COO(
            [[0]*len(token_indices), token_indices],
            [1.0]*len(token_indices),
            shape=(1, len(self.token2idx))
        )

        return vec

    def train(
            self,
            classdict: dict[str, list[str]],
            nb_epochs: int = 500,
            l2reg: float = 0.01,
            bias_l2reg: float = 0.01,
            optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam"
    ) -> None:
        """ Train the classifier.

        Given the training data, train the classifier.

        :param classdict: training data
        :param nb_epochs: number of epochs (Defauly: 500)
        :param l2reg: L2 regularization coefficient (Default: 0.01)
        :param bias_l2reg: L2 regularization coefficient for bias (Default: 0.01)
        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
        :return: None
        :type classdict: dict
        :type nb_epochs: int
        :type l2reg: float
        :type bias_l2reg: float
        :type optimizer: str
        """
        self.classlabels = sorted(classdict.keys())
        self.labels2idx = {label: idx for idx, label in enumerate(self.classlabels)}

        dtm_npdict_matrix, y = convert_classdict_to_xy(
            classdict, self.labels2idx, preprocess_func=self.preprocess_func, tokenize_func=tokenize
        )
        self.token2idx = {
            token: idx
            for idx, token in enumerate(dtm_npdict_matrix._lists_keystrings[1])
        }

        kmodel = logistic_framework(
            dtm_npdict_matrix.dimension_sizes[1],
            len(self.classlabels),
            l2reg=l2reg,
            bias_l2reg=bias_l2reg,
            optimizer=optimizer
        )
        kmodel.fit(dtm_npdict_matrix.to_numpy(), y.todense(), epochs=nb_epochs)

        self.model = kmodel
        self.trained = True

    def savemodel(self, nameprefix: str) -> None:
        """ Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There will be give files produced, one name ending with "_classlabels.txt", one with ".json",
        one with ".weights.h5", one with "_labelidx.pkl", and one with "_dictionary.dict".

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        kerasio.save_model(nameprefix, self.model)
        open(nameprefix+'_tokens2idx.json', 'wb').write(orjson.dumps(self.token2idx))
        open(nameprefix+'_classlabels.txt', 'w').write('\n'.join(self.classlabels))
        open(nameprefix+'_labels2idx.json', 'wb').write(orjson.dumps(self.labels2idx))

    def loadmodel(self, nameprefix: str) -> None:
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_classlabels.txt", ".json", ".weights.h5", "_labelidx.pkl", and "_dictionary.dict".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving the model.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        """
        self.model = kerasio.load_model(nameprefix)
        self.token2idx = orjson.loads(open(nameprefix+"_tokens2idx.json", "rb").read())
        self.classlabels = [
            s.strip()
            for s in open(nameprefix+'_classlabels.txt', 'r').readlines()
        ]
        self.labels2idx = orjson.loads(open(nameprefix+"_labels2idx.json", "rb").read())
        self.trained = True

    def score(self, shorttext: str) -> dict[str, float]:
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
        if not self.trained:
            raise e.ModelNotTrainedException()

        vec = self.shorttext_to_vec(shorttext)
        predictions = self.model.predict(vec.todense())

        # wrangle output result
        scoredict = {
            classlabel: predictions[0][idx]
            for idx, classlabel in enumerate(self.classlabels)
        }
        return scoredict


def load_maxent_classifier(name: str, compact: bool=True) -> MaxEntClassifier:
    """ Load the maximum entropy classifier from saved model.

    Given a model file(s), load the maximum entropy classifier.

    :param name: name or prefix of the file, if compact is True or False respectively
    :param compact: whether the model file is compact (Default:True)
    :return: maximum entropy classifier
    :type name: str
    :type compact: bool
    :rtype: MaxEntClassifier
    """
    classifier = MaxEntClassifier()
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier