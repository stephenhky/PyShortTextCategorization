
import pickle
from abc import ABC, abstractmethod
from typing import Optional, Annotated, Generator, Literal

import numpy as np
import numpy.typing as npt
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from ..utils.classification_exceptions import ModelNotTrainedException
from ..utils import kerasmodel_io as kerasio
from ..utils.compactmodel_io import CompactIOMachine
from ..classifiers.base import AbstractScorer


# abstract class
class StackedGeneralization(ABC):
    """Abstract base class for stacked generalization.

    An intermediate model that takes output from other classifiers as input
    features and performs another level of classification.

    The classifiers must have the :meth:`~score` method that takes a string as input.

    Reference:
        David H. Wolpert, "Stacked Generalization," Neural Netw 5: 241-259 (1992).

        M. Paz Sesmero et al., "Generating ensembles of heterogeneous classifiers
        using Stacked Generalization," WIREs Data Mining and Knowledge Discovery 5: 21-34 (2015).
    """

    def __init__(
            self,
            intermediate_classifiers: Optional[dict[str, AbstractScorer]] = None
    ):
        """Initialize the stacking class.

        Args:
            intermediate_classifiers: Dictionary mapping names to classifier instances.
        """
        self.classifiers = intermediate_classifiers if intermediate_classifiers is not None else {}
        self.classlabels = []
        self.trained = False

    def register_classifiers(self) -> None:
        """Register the intermediate classifiers.

        Must be called before training.
        """
        self.classifier2idx = {}
        self.idx2classifier = {}
        for idx, key in enumerate(self.classifiers.keys()):
            self.classifier2idx[key] = idx
            self.idx2classifier[idx] = key

    def register_classlabels(self, labels: list[str]) -> None:
        """Register output labels.

        Args:
            labels: List of output class labels.

        Must be called before training.
        """
        self.classlabels = labels
        self.labels2idx = {classlabel: idx for idx, classlabel in enumerate(self.classlabels)}

    def add_classifier(self, name: str, classifier: AbstractScorer) -> None:
        """Add a classifier to the stack.

        Args:
            name: Name for the classifier (no spaces or special characters).
            classifier: Classifier instance with a :meth:`~score` method.
        """
        self.classifiers[name] = classifier
        self.register_classifiers()

    def delete_classifier(self, name: str) -> None:
        """Delete a classifier from the stack.

        Args:
            name: Name of the classifier to delete.

        Raises:
            KeyError: If classifier name not found.
        """
        del self.classifiers[name]
        self.register_classifiers()

    def translate_shorttext_intfeature_matrix(
            self,
            shorttext: str
    ) -> Annotated[npt.NDArray[np.float64], "2D Array"]:
        """Convert short text to feature matrix for stacking.

        Args:
            shorttext: Input text.

        Returns:
            Feature matrix of shape (n_classifiers, n_labels).
        """
        feature_matrix = np.zeros((len(self.classifier2idx), len(self.labels2idx)))
        for key, idx in self.classifier2idx.items():
            classifier = self.classifiers[key]
            scoredict = classifier.score(shorttext)
            for label in scoredict:
                feature_matrix[idx, self.labels2idx[label]] = scoredict[label]
        return feature_matrix

    def convert_label_to_buckets(
            self,
            label: str
    ) -> Annotated[npt.NDArray[np.int64], "1D Array"]:
        """Convert label to one-hot bucket representation.

        Args:
            label: Class label.

        Returns:
            One-hot array with 1 at the label's position.
        """
        buckets = np.zeros(len(self.labels2idx), dtype=np.int64)
        buckets[self.labels2idx[label]] = 1
        return buckets

    def convert_traindata_matrix(
            self,
            classdict: dict[str, list[str]],
            tobucket: bool = True
    ) -> Generator[tuple[Annotated[npt.NDArray[np.float64], "2D Array"], Annotated[npt.NDArray[np.int64], "1D Array"]], None, None]:
        """Yield training data matrices.

        Args:
            classdict: Training data dictionary.
            tobucket: Whether to convert labels to buckets. Default: True.

        Yields:
            Tuples of (feature_matrix, label_array).
        """
        for label, texts in classdict.items():
            y = self.convert_label_to_buckets(label) if tobucket else self.labels2idx[label]
            for shorttext in texts:
                x = self.translate_shorttext_intfeature_matrix(shorttext)
                yield x, y

    @abstractmethod
    def train(self, classdict: dict[str, list[str]], *args, **kwargs) -> None:
        """Train the stacked generalization model.

        Args:
            classdict: Training data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()

    @abstractmethod
    def score(self, shorttext: str, *args, **kwargs) -> dict[str, float]:
        """Calculate classification scores for all labels.

        Args:
            shorttext: Input text.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            NotImplementedError: Abstract method.
        """
        raise NotImplemented()


class LogisticStackedGeneralization(StackedGeneralization, CompactIOMachine):
    """Stacked generalization using logistic regression.

    Uses neural network with sigmoid output to combine predictions from
    intermediate classifiers.

    Note:
        Saves the stacked model but not the intermediate classifiers.
    """

    def __init__(
            self,
            intermediate_classifiers: Optional[dict[str, AbstractScorer]] = None,
    ):
        CompactIOMachine.__init__(self,
                                  {'classifier': 'stacked_logistics'},
                                  'stacked_logistics',
                                  ['_stackedlogistics.pkl', '_stackedlogistics.weights.h5', '_stackedlogistics.json'])
        StackedGeneralization.__init__(self, intermediate_classifiers=intermediate_classifiers)

    def train(
            self,
            classdict: dict[str, list[str]],
            optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam",
            l2reg: float = 0.01,
            bias_l2reg: float = 0.01,
            nb_epoch: int = 1000
    ) -> None:
        """Train the stacked generalization model.

        Args:
            classdict: Training data.
            optimizer: Optimizer for training. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. Default: adam.
            l2reg: L2 regularization coefficient. Default: 0.01.
            bias_l2reg: L2 regularization for bias. Default: 0.01.
            nb_epoch: Number of training epochs. Default: 1000.
        """

        # register
        self.register_classifiers()
        self.register_classlabels(sorted(classdict.keys()))    # sorted the keys

        kmodel = Sequential()
        kmodel.add(Reshape((len(self.classifier2idx) * len(self.labels2idx),),
                           input_shape=(len(self.classifier2idx), len(self.labels2idx))))
        kmodel.add(Dense(units=len(classdict),
                         activation='sigmoid',
                         kernel_regularizer=l2(l2reg),
                         bias_regularizer=l2(bias_l2reg))
                   )
        kmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)

        Xy = [(xone, yone) for xone, yone in self.convert_traindata_matrix(classdict, tobucket=True)]
        X = np.array([item[0] for item in Xy])
        y = np.array([item[1] for item in Xy])

        kmodel.fit(X, y, epochs=nb_epoch)

        self.model = kmodel
        self.trained = True

    def score(self, shorttext: str) -> dict[str, float]:
        """Calculate classification scores for all labels.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()

        input_matrix = self.translate_shorttext_intfeature_matrix(shorttext)
        prediction = self.model.predict(np.array([input_matrix]))

        scoredict = {label: prediction[0][idx] for idx, label in enumerate(self.classlabels)}

        return scoredict

    def savemodel(self, nameprefix: str) -> None:
        """Save the stacked model to files.

        Note: Intermediate classifiers are not saved. Save them separately.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()

        stackedmodeldict = {'classifiers': self.classifier2idx,
                            'classlabels': self.classlabels}
        pickle.dump(stackedmodeldict, open(nameprefix+'_stackedlogistics.pkl', 'wb'))
        kerasio.save_model(nameprefix+'_stackedlogistics', self.model)

    def loadmodel(self, nameprefix: str) -> None:
        """Load the stacked model from files.

        Note: Intermediate classifiers are not loaded. Load them separately.

        Args:
            nameprefix: Prefix for input files.
        """
        stackedmodeldict = pickle.load(open(nameprefix+'_stackedlogistics.pkl', 'rb'))
        self.register_classlabels(stackedmodeldict['classlabels'])
        self.classifier2idx = stackedmodeldict['classifiers']
        self.idx2classifier = {val: key for key, val in self.classifier2idx.items()}
        self.model = kerasio.load_model(nameprefix+'_stackedlogistics')

        self.trained = True
