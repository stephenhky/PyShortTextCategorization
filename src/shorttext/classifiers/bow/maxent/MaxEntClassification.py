
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
    """Create a maximum entropy classifier neural network.

    Args:
        nb_features: Number of input features.
        nb_outputs: Number of output classes.
        l2reg: L2 regularization coefficient. Default: 0.01.
        bias_l2reg: L2 regularization for bias. Default: 0.01.
        optimizer: Optimizer. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. Default: adam.

    Returns:
        Keras Sequential model for maximum entropy classification.
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
    """Maximum entropy classifier.

    A classifier that implements the principle of maximum entropy
    for text categorization using bag-of-words features.

    Reference:
        Adam L. Berger et al., "A Maximum Entropy Approach to Natural
        Language Processing," Computational Linguistics 22(1): 39-72 (1996).
    """

    def __init__(self, preprocessor: Optional[callable] = None):
        """Initialize the classifier.

        Args:
            preprocessor: Text preprocessing function. Default: lowercase.
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
        """Convert short text to sparse vector.

        Args:
            shorttext: Input text.

        Returns:
            Sparse vector representation.
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
        """Train the classifier.

        Args:
            classdict: Training data.
            nb_epochs: Number of training epochs. Default: 500.
            l2reg: L2 regularization coefficient. Default: 0.01.
            bias_l2reg: L2 regularization for bias. Default: 0.01.
            optimizer: Optimizer. Default: adam.
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
        """Save the trained model to files.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        kerasio.save_model(nameprefix, self.model)
        open(nameprefix+'_tokens2idx.json', 'wb').write(orjson.dumps(self.token2idx))
        open(nameprefix+'_classlabels.txt', 'w').write('\n'.join(self.classlabels))
        open(nameprefix+'_labels2idx.json', 'wb').write(orjson.dumps(self.labels2idx))

    def loadmodel(self, nameprefix: str) -> None:
        """Load a trained model from files.

        Args:
            nameprefix: Prefix for input files.
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
        """Calculate classification scores for all class labels.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            ModelNotTrainedException: If not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        vec = self.shorttext_to_vec(shorttext)
        predictions = self.model.predict(vec.todense())

        scoredict = {
            classlabel: predictions[0][idx]
            for idx, classlabel in enumerate(self.classlabels)
        }
        return scoredict


def load_maxent_classifier(name: str, compact: bool=True) -> MaxEntClassifier:
    """Load a MaxEntClassifier from file.

    Args:
        name: Model name (compact) or file prefix (non-compact).
        compact: Whether to load compact model. Default: True.

    Returns:
        MaxEntClassifier instance.
    """
    classifier = MaxEntClassifier()
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier