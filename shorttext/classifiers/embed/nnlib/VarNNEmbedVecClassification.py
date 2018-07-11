
import json
import os
import warnings

import numpy as np

import shorttext.utils.kerasmodel_io as kerasio
import shorttext.utils.classification_exceptions as e
from shorttext.utils import tokenize
from shorttext.utils.compactmodel_io import CompactIOMachine


class VarNNEmbeddedVecClassifier(CompactIOMachine):
    """
    This is a wrapper for various neural network algorithms
    for supervised short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    The sentences are represented by a matrix, or rank-2 array.
    The type of neural network has to be passed when training, and it has to be of
    type :class:`keras.models.Sequential`. The number of outputs of the models has to match
    the number of class labels in the training data.
    To perform prediction, the input short sentences is converted to a unit vector
    in the same way. The score is calculated according to the trained neural network model.

    Examples of the models can be found in `frameworks`.

    A pre-trained Google Word2Vec model can be downloaded `here
    <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.
    """
    def __init__(self, wvmodel, vecsize=None, maxlen=15, with_gensim=False):
        """ Initialize the classifier.

        :param wvmodel: Word2Vec model
        :param vecsize: length of the embedded vectors in the model (Default: None, directly extracted from word-embedding model)
        :param maxlen: maximum number of words in a sentence (Default: 15)
        :type wvmodel: gensim.models.keyedvectors.KeyedVectors
        :type vecsize: int
        :type maxlen: int
        """
        CompactIOMachine.__init__(self, {'classifier': 'nnlibvec'}, 'nnlibvec', ['_classlabels.txt', '.json', '.h5', '_config.json'])
        self.wvmodel = wvmodel
        self.vecsize = self.wvmodel.vector_size if vecsize == None else vecsize
        self.maxlen = maxlen
        self.with_gensim = False if not with_gensim else with_gensim
        self.trained = False

    def convert_trainingdata_matrix(self, classdict):
        """ Convert the training data into format put into the neural networks.

        Convert the training data into format put into the neural networks.
        This is called by :func:`~train`.

        :param classdict: training data
        :return: a tuple of three, containing a list of class labels, matrix of embedded word vectors, and corresponding outputs
        :type classdict: dict
        :rtype: (list, numpy.ndarray, list)
        """
        classlabels = classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        # tokenize the words, and determine the word length
        phrases = []
        indices = []
        for label in classlabels:
            for shorttext in classdict[label]:
                shorttext = shorttext if type(shorttext)==str else ''
                category_bucket = [0]*len(classlabels)
                category_bucket[lblidx_dict[label]] = 1
                indices.append(category_bucket)
                phrases.append(tokenize(shorttext))

        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)

        return classlabels, train_embedvec, indices

    def train(self, classdict, kerasmodel, nb_epoch=10):
        """ Train the classifier.

        The training data and the corresponding keras model have to be given.

        If this has not been run, or a model was not loaded by :func:`~loadmodel`,
        a `ModelNotTrainedException` will be raised.

        :param classdict: training data
        :param kerasmodel: keras sequential model
        :param nb_epoch: number of steps / epochs in training
        :return: None
        :type classdict: dict
        :type kerasmodel: keras.models.Sequential
        :type nb_epoch: int
        """
        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix(classdict)

        # train the model
        kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

    def savemodel(self, nameprefix):
        """ Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There will be three files produced, one name ending with "_classlabels.txt", one name
        ending with ".json", and one name ending with ".h5". For shorttext>=0.4.0, another file
        with extension "_config.json" would be created.

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        kerasio.save_model(nameprefix, self.model)
        labelfile = open(nameprefix+'_classlabels.txt', 'w')
        labelfile.write('\n'.join(self.classlabels))
        labelfile.close()
        json.dump({'with_gensim': False, 'maxlen': self.maxlen, 'vecsize': self.vecsize},
                  open(nameprefix+'_config.json', 'w'))

    def loadmodel(self, nameprefix):
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_classlabels.txt", ".json" and ".h5". For shorttext>=0.4.0, a file with
        extension "_config.json" would also be used.

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving the model.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        """
        self.model = kerasio.load_model(nameprefix)
        labelfile = open(nameprefix+'_classlabels.txt', 'r')
        self.classlabels = labelfile.readlines()
        labelfile.close()
        self.classlabels = [s.strip() for s in self.classlabels]

        # check if _config.json exists.
        # This file does not exist if the model was created with shorttext<0.4.0
        if os.path.exists(nameprefix+'_config.json'):
            config = json.load(open(nameprefix+'_config.json', 'r'))
            # these fields are present for release >= 1.0.0
            if 'maxlen' in config:
                self.maxlen = config['maxlen']
            else:
                self.maxlen = 15
            if 'vecsize' in config:
                self.vecsize = config['vecsize']
            else:
                self.vecsize = self.wvmodel.vector_size
            if self.vecsize != self.wvmodel.vector_size:
                warnings.warn('Record vector size (%i) is not the same as that of the given word-embedding model (%i)! ' % (self.vecsize, self.wvmodel.vector_size)+
                              'Setting the vector size to be %i, but there may be run time error!' % (self.wvmodel.vector_size),
                              RuntimeWarning)
                self.vecsize = self.wvmodel.vector_size
        else:
            self.maxlen = 15
            self.vecsize = self.wvmodel.vector_size
            warnings.warn('Model files from old versions.')

        self.with_gensim = False
        self.trained = True

    def word_to_embedvec(self, word):
        """ Convert the given word into an embedded vector.

        Given a word, return the corresponding embedded vector according to
        the word-embedding model. If there is no such word in the model,
        a vector with zero values are given.

        :param word: a word
        :return: the corresponding embedded vector
        :type word: str
        :rtype: numpy.ndarray
        """
        return self.wvmodel[word] if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_matrix(self, shorttext):
        """ Convert the short text into a matrix with word-embedding representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, and put them into a matrix. If a word is not in the model,
        that row will be filled with zero.

        :param shorttext: a short sentence
        :return: a matrix of embedded vectors that represent all the tokens in the sentence
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        tokens = tokenize(shorttext)
        matrix = np.zeros((self.maxlen, self.vecsize))
        for i in range(min(self.maxlen, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix

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

        # retrieve vector
        matrix = np.array([self.shorttext_to_matrix(shorttext)])

        # classification using the neural network
        predictions = self.model.predict(matrix)

        # wrangle output result
        scoredict = {classlabel: predictions[0][idx]
                     for idx, classlabel in zip(range(len(self.classlabels)), self.classlabels)}

        return scoredict


def load_varnnlibvec_classifier(wvmodel, name, compact=True, vecsize=None):
    """ Load a :class:`shorttext.classifiers.VarNNEmbeddedVecClassifier` instance from file, given the pre-trained word-embedding model.

    :param wvmodel: Word2Vec model
    :param name: name (if compact=True) or prefix (if compact=False) of the file path
    :param compact whether model file is compact (Default: True)
    :param vecsize: length of embedded vectors in the model (Default: None, extracted directly from the word-embedding model)
    :return: the classifier
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type name: str
    :type compact: bool
    :type vecsize: int
    :rtype: VarNNEmbeddedVecClassifier
    """
    classifier = VarNNEmbeddedVecClassifier(wvmodel, vecsize=vecsize)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier
