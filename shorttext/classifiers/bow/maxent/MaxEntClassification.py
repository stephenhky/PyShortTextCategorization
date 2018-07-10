
import pickle

from scipy.sparse import dok_matrix
from gensim.corpora import Dictionary
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

import shorttext.utils.kerasmodel_io as kerasio
from shorttext.utils import tokenize
from shorttext.utils import gensim_corpora as gc
from shorttext.utils import classification_exceptions as e
from shorttext.utils.compactmodel_io import CompactIOMachine
from shorttext.utils import deprecated


def logistic_framework(nb_features, nb_outputs, l2reg=0.01, bias_l2reg=0.01, optimizer='adam'):
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


class MaxEntClassifier(CompactIOMachine):
    """
    This is a classifier that implements the principle of maximum entropy.

    Reference:
    * Adam L. Berger, Stephen A. Della Pietra, Vincent J. Della Pietra, "A Maximum Entropy Approach to Natural Language Processing," *Computational Linguistics* 22(1): 39-72 (1996).
    """
    def __init__(self, preprocessor=lambda s: s.lower()):
        """ Initializer.

        :param preprocessor: text preprocessor
        :type preprocessor: function
        """
        CompactIOMachine.__init__(self,
                                  {'classifier': 'maxent'},
                                  'maxent',
                                  ['_classlabels.txt', '.json', '.h5', '_labelidx.pkl', '_dictionary.dict'])
        self.preprocessor = preprocessor
        self.trained = False

    def shorttext_to_vec(self, shorttext):
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
        # too slow, deprecated
        tokens = tokenize(self.preprocessor(shorttext))

        vec = dok_matrix((1, len(self.dictionary)))
        for token in tokens:
            if token in self.dictionary.token2id:
                vec[0, self.dictionary.token2id[token]] = 1.0

        return vec[0, :]

    @deprecated
    def gensimcorpus_to_matrix(self, corpus):
        """ Convert the gensim corpus into a sparse matrix. (deprecated)

        :param corpus: gensim corpus
        :return: matrix representing the corpus
        :type corpus: list
        :rtype: scipy.sparse.dok_matrix
        """
        # not used, deprecated
        matrix = dok_matrix((len(corpus), len(self.dictionary)))
        for docid, doc in enumerate(corpus):
            for tokenid, count in doc:
                matrix[docid, tokenid] = count
        return matrix

    def index_classlabels(self):
        """ Index the class outcome labels.

        Index the class outcome labels into integers, for neural network implementation.

        """
        self.labels2idx = {label: idx for idx, label in enumerate(self.classlabels)}

    def convert_classdict_to_XY(self, classdict):
        """ Convert the training data into sparse matrices for training.

        :param classdict: training data
        :return: a tuple, consisting of sparse matrices for X (training data) and y (the labels of the training data)
        :type classdict: dict
        :rtype: tuple
        """
        nb_data = sum([len(classdict[k]) for k in classdict])
        X = dok_matrix((nb_data, len(self.dictionary)))
        y = dok_matrix((nb_data, len(self.labels2idx)))

        rowid = 0
        for label in classdict:
            if label in self.labels2idx.keys():
                for shorttext in classdict[label]:
                    tokens = tokenize(self.preprocessor(shorttext))
                    #X[rowid, :] = self.shorttext_to_vec(shorttext)
                    for token in tokens:
                        X[rowid, self.dictionary.token2id[token]] += 1.0
                    y[rowid, self.labels2idx[label]] = 1.
                    rowid += 1

        return X, y

    def train(self, classdict, nb_epochs=500, l2reg=0.01, bias_l2reg=0.01, optimizer='adam'):
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
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda s: tokenize(self.preprocessor(s)))
        self.index_classlabels()

        X, y = self.convert_classdict_to_XY(classdict)

        kmodel = logistic_framework(len(self.dictionary),
                                    len(self.classlabels),
                                    l2reg=l2reg,
                                    bias_l2reg=bias_l2reg,
                                    optimizer=optimizer)
        kmodel.fit(X.toarray(), y.toarray(), epochs=nb_epochs)

        self.model = kmodel
        self.trained = True

    def savemodel(self, nameprefix):
        """ Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There will be give files produced, one name ending with "_classlabels.txt", one with ".json",
        one with ".h5", one with "_labelidx.pkl", and one with "_dictionary.dict".

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        kerasio.save_model(nameprefix, self.model)

        self.dictionary.save(nameprefix+'_dictionary.dict')

        labelfile = open(nameprefix+'_classlabels.txt', 'w')
        labelfile.write('\n'.join(self.classlabels))
        labelfile.close()

        pickle.dump(self.labels2idx, open(nameprefix+'_labelidx.pkl', 'wb'))

    def loadmodel(self, nameprefix):
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_classlabels.txt", ".json", ".h5", "_labelidx.pkl", and "_dictionary.dict".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving the model.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        """
        self.model = kerasio.load_model(nameprefix)

        self.dictionary = Dictionary.load(nameprefix+'_dictionary.dict')

        labelfile = open(nameprefix+'_classlabels.txt', 'r')
        self.classlabels = [s.strip() for s in labelfile.readlines()]
        labelfile.close()

        self.labels2idx = pickle.load(open(nameprefix+'_labelidx.pkl', 'rb'))

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
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        vec = self.shorttext_to_vec(shorttext)
        predictions = self.model.predict(vec.toarray())

        # wrangle output result
        scoredict = {classlabel: predictions[0][idx] for idx, classlabel in enumerate(self.classlabels)}
        return scoredict


def load_maxent_classifier(name, compact=True):
    """ Load the maximum entropy classifier from saved model.

    Given a moel file(s), load the maximum entropy classifier.

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