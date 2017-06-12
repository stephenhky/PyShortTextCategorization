import numpy as np

import shorttext.utils.kerasmodel_io as kerasio
import shorttext.utils.classification_exceptions as e
from shorttext.utils.textpreprocessing import spacy_tokenize


class VarNNSumEmbeddedVecClassifier:
    """
    This is a wrapper for various neural network algorithms
    for supervised short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    The sentences are represented by an array.
    The type of neural network has to be passed when training, and it has to be of
    type :class:`keras.models.Sequential`. The number of outputs of the models has to match
    the number of class labels in the training data.
    To perform prediction, the input short sentences is converted to a unit vector
    in the same way. The score is calculated according to the trained neural network model.

    Examples of the models can be found in `frameworks`.

    A pre-trained Google Word2Vec model can be downloaded `here
    <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.

    """
    def __init__(self, wvmodel, vecsize=300, maxlen=15):
        """ Initialize the classifier.

        :param wvmodel: Word2Vec model
        :param vecsize: length of the embedded vectors in the model (Default: 300)
        :param maxlen: maximum number of words in a sentence (Default: 15)
        :type wvmodel: gensim.models.word2vec.Word2Vec
        :type vecsize: int
        :type maxlen: int
        """
        self.wvmodel = wvmodel
        self.vecsize = vecsize
        self.maxlen = maxlen
        self.trained = False

    def convert_traindata_embedvecs(self, classdict):
        """ Convert the training text data into embedded matrix.

        COnvert the training text data into embedded matrix, where each short sentence
        is a normalized summed embedded vectors for all words.

        :param classdict: training data
        :return: tuples, consisting of class labels, matrix of embedded vectors, and corresponding outputs
        :type classdict: dict
        :rtype: (list, numpy.ndarray, list)
        """
        classlabels = classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        indices = []
        embedvecs = []
        for classlabel in classlabels:
            for shorttext in classdict[classlabel]:
                embedvec = np.sum(np.array([self.word_to_embedvec(token) for token in spacy_tokenize(shorttext)]),
                                  axis=0)
                # embedvec = np.reshape(embedvec, embedvec.shape+(1,))
                norm = np.linalg.norm(embedvec)
                if norm == 0:
                    continue
                embedvec /= norm
                embedvecs.append(embedvec)
                category_bucket = [0]*len(classlabels)
                category_bucket[lblidx_dict[classlabel]] = 1
                indices.append(category_bucket)

        indices = np.array(indices)
        embedvecs = np.array(embedvecs)
        return classlabels, embedvecs, indices

    def train(self, classdict, kerasmodel, nb_epoch=10):
        """ Train the classifier.

        The training data and the corresponding keras model have to be given.

        If this has not been run, or a model was not loaded by :func:`~loadmodel`,
        a `ModelNotTrainedException` will be raised while performing prediction and saving the model.

        :param classdict: training data
        :param kerasmodel: keras sequential model
        :param nb_epoch: number of steps / epochs in training
        :return: None
        :type classdict: dict
        :type kerasmodel: keras.models.Sequential
        :type nb_epoch: int
        """
        # convert training data into embedded vectors
        self.classlabels, train_embedvec, indices = self.convert_traindata_embedvecs(classdict)

        # train the model
        kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

    def savemodel(self, nameprefix):
        """ Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix.
        There will be three files produced, one name ending with "_classlabels.txt", one name
        ending with ".json", and one name ending with ".h5".
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

    def loadmodel(self, nameprefix):
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_classlabels.txt", ".json", and ".h5".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction and saving the model.

        :param nameprefix: prefix of the file path
        :return: None
        :type nameprefix: str
        """
        self.model = kerasio.load_model(nameprefix)
        labelfile = open(nameprefix+'_classlabels.txt', 'r')
        self.classlabels = labelfile.readlines()
        labelfile.close()
        self.classlabels = map(lambda s: s.strip(), self.classlabels)
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

    def shorttext_to_embedvec(self, shorttext):
        """ Convert the short text into an averaged embedded vector representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, sums
        them up, and normalize the resulting vector. It returns the resulting vector
        that represents this short sentence.

        :param shorttext: a short sentence
        :return: an embedded vector that represents the short sentence
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        vec = np.zeros(self.vecsize)
        for token in spacy_tokenize(shorttext):
            if token in self.wvmodel:
                vec += self.wvmodel[token]
        norm = np.linalg.norm(vec)
        if norm!=0:
            vec /= np.linalg.norm(vec)
        return vec

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
        embedvec = np.array(self.shorttext_to_embedvec(shorttext))

        # classification using the neural network
        predictions = self.model.predict(np.array([embedvec]))

        # wrangle output result
        scoredict = {classlabel: predictions[0][idx] for idx, classlabel in enumerate(self.classlabels)}
        return scoredict