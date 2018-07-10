import pickle

import numpy as np
from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.regularizers import l2

import shorttext.utils.classification_exceptions as e
import shorttext.utils.kerasmodel_io as kerasio
from shorttext.utils.compactmodel_io import CompactIOMachine


# abstract class
class StackedGeneralization:
    """
    This is an abstract class for any stacked generalization method. It is an intermediate model
    that takes the results of other classifiers as the input features, and perform another classification.

    The classifiers must have the :func:`~score` method that takes a string as an input argument.

    More references:

    David H. Wolpert, "Stacked Generalization," *Neural Netw* 5: 241-259 (1992).

    M. Paz Sesmero, Agapito I. Ledezma, Araceli Sanchis, "Generating ensembles of heterogeneous classifiers using Stacked Generalization,"
    *WIREs Data Mining and Knowledge Discovery* 5: 21-34 (2015).
    """
    def __init__(self, intermediate_classifiers={}):
        """ Initialize the stacking class instance.

        :param intermediate_classifiers: dictionary, with key being a string, and the values intermediate classifiers, that have the method :func:`~score`, which takes a string as the input argument.
        :type intermediate_classifiers: dict
        """
        self.classifiers = intermediate_classifiers
        self.classlabels = []
        self.trained = False

    def register_classifiers(self):
        """ Register the intermediate classifiers.

        It must be run before any training.

        :return: None
        """
        self.classifier2idx = {}
        self.idx2classifier = {}
        for idx, key in enumerate(self.classifiers.keys()):
            self.classifier2idx[key] = idx
            self.idx2classifier[idx] = key

    def register_classlabels(self, labels):
        """ Register output labels.

        Given the labels, it gives an integer as the index for each label.
        It is essential for the output model to place.

        It must be run before any training.

        :param labels: list of output labels
        :return: None
        :type labels: list
        """
        self.classlabels = list(labels)
        self.labels2idx = {classlabel: idx for idx, classlabel in enumerate(self.classlabels)}

    def add_classifier(self, name, classifier):
        """ Add a classifier.

        Add a classifier to the class. The classifier must have the method :func:`~score` which
        takes a string as an input argument.

        :param name: name of the classifier, without spaces and any special characters
        :param classifier: instance of a classifier, which has a method :func:`~score` which takes a string as an input argument
        :return: None
        :type name: str
        :type classifier: any class with a method :func:`~score`
        """
        self.classifiers[name] = classifier
        self.register_classifiers()

    def delete_classifier(self, name):
        """ Delete a classifier.

        :param name: name of the classifier to be deleted
        :return: None
        :type name: str
        :raise: KeyError
        """
        del self.classifiers[name]
        self.register_classifiers()

    def translate_shorttext_intfeature_matrix(self, shorttext):
        """ Represent the given short text as the input matrix of the stacking class.

        :param shorttext: short text
        :return: input matrix of the stacking class
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        feature_matrix = np.zeros((len(self.classifier2idx), len(self.labels2idx)))
        for key in self.classifier2idx:
            scoredict = self.classifiers[key].score(shorttext)
            for label in scoredict:
                feature_matrix[self.classifier2idx[key], self.labels2idx[label]] = scoredict[label]
        return feature_matrix

    def convert_label_to_buckets(self, label):
        """ Convert the label into an array of bucket.

        Some classification algorithms, especially those of neural networks, have the output
        as a serious of buckets with the correct answer being 1 in the correct label, with other being 0.
        This method convert the label into the corresponding buckets.

        :param label: label
        :return: array of buckets
        :type label: str
        :rtype: numpy.ndarray
        """
        buckets = np.zeros(len(self.labels2idx), dtype=np.int)
        buckets[self.labels2idx[label]] = 1
        return buckets

    def convert_traindata_matrix(self, classdict, tobucket=True):
        """ Returns a generator that returns the input matrix and the output labels for training.

        :param classdict: dictionary of the training data
        :param tobucket: whether to convert the label into buckets (Default: True)
        :return: array of input matrix, and output labels
        :type classdict: dict
        :type tobucket: bool
        :rtype: tuple
        """
        for label in classdict:
            y = self.convert_label_to_buckets(label) if tobucket else self.labels2idx[label]
            for shorttext in classdict[label]:
                X = self.translate_shorttext_intfeature_matrix(shorttext)
                yield X, y

    def train(self, classdict, *args, **kwargs):
        """ Train the stacked generalization.

        Not implemented. `NotImplemntedException` raised.

        :param classdict: training data
        :param args: arguments to be parsed
        :param kwargs: arguments to be parsed
        :return: None
        :type classdict: dict
        :type args: dict
        :type kwargs: dict
        :raise: NotImplementedException
        """
        raise e.NotImplementedException()

    def score(self, shorttext, *args, **kwargs):
        """ Calculate the scores for each class labels.

        Not implemented. `NotImplemntedException` raised.

        :param shorttext: short text to be scored
        :param args: arguments to be parsed
        :param kwargs: arguments to be parsed
        :return: dictionary of scores for all class labels
        :type shorttext: str
        :type args: dict
        :type kwargs: dict
        :rtype: dict
        :raise: NotImplementedException
        """
        raise e.NotImplementedException()


class LogisticStackedGeneralization(StackedGeneralization, CompactIOMachine):
    """
    This class implements logistic regression as the stacked generalizer.

    It is an intermediate model
    that takes the results of other classifiers as the input features, and perform another classification.

    This class saves the stacked logistic model, but not the information of the primary model.

    The classifiers must have the :func:`~score` method that takes a string as an input argument.
    """
    def __init__(self, intermediate_classifiers={}):
        CompactIOMachine.__init__(self,
                                  {'classifier': 'stacked_logistics'},
                                  'stacked_logistics',
                                  ['_stackedlogistics.pkl', '_stackedlogistics.h5', '_stackedlogistics.json'])
        StackedGeneralization.__init__(self, intermediate_classifiers=intermediate_classifiers)

    def train(self, classdict, optimizer='adam', l2reg=0.01, bias_l2reg=0.01, nb_epoch=1000):
        """ Train the stacked generalization.

        :param classdict: training data
        :param optimizer: optimizer to use Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: 'adam', for adam optimizer)
        :param l2reg: coefficients for L2-regularization (Default: 0.01)
        :param bias_l2reg: coefficients for L2-regularization for bias (Default: 0.01)
        :param nb_epoch: number of epochs for training (Default: 1000)
        :return: None
        :type classdict: dict
        :type optimizer: str
        :type l2reg: float
        :type bias_l2reg: float
        :type nb_epoch: int
        """

        # register
        self.register_classifiers()
        self.register_classlabels(classdict.keys())

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

    def score(self, shorttext):
        """ Calculate the scores for all the class labels for the given short sentence.

        Given a short sentence, calculate the classification scores for all class labels,
        returned as a dictionary with key being the class labels, and values being the scores.
        If the short sentence is empty, or if other numerical errors occur, the score will be `numpy.nan`.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: a short sentence
        :return: a dictionary with keys being the class labels, and values being the corresponding classification scores
        :type shorttext: str
        :rtype: dict
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        input_matrix = self.translate_shorttext_intfeature_matrix(shorttext)
        prediction = self.model.predict(np.array([input_matrix]))

        scoredict = {label: prediction[0][idx] for idx, label in enumerate(self.classlabels)}

        return scoredict

    def savemodel(self, nameprefix):
        """ Save the logistic stacked model into files.

        Save the stacked model into files. Note that the intermediate classifiers
        are not saved. Users are advised to save those classifiers separately.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param nameprefix: prefix of the files
        :return: None
        :raise: ModelNotTrainedException
        :type nameprefix: str
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        stackedmodeldict = {'classifiers': self.classifier2idx,
                            'classlabels': self.classlabels}
        pickle.dump(stackedmodeldict, open(nameprefix+'_stackedlogistics.pkl', 'wb'))
        kerasio.save_model(nameprefix+'_stackedlogistics', self.model)

    def loadmodel(self, nameprefix):
        """ Load the model with the given prefix.

        Load the model with the given prefix of their paths. Note that the intermediate
        classifiers are not loaded, and users are required to load them separately.

        :param nameprefix: prefix of the model files
        :return: None
        :type nameprefix: str
        """
        stackedmodeldict = pickle.load(open(nameprefix+'_stackedlogistics.pkl', 'rb'))
        self.register_classlabels(stackedmodeldict['classlabels'])
        self.classifier2idx = stackedmodeldict['classifiers']
        self.idx2classifier = {val: key for key, val in self.classifier2idx.items()}
        self.model = kerasio.load_model(nameprefix+'_stackedlogistics')

        self.trained = True




