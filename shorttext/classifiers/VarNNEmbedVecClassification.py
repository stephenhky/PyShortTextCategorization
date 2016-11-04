import numpy as np
from nltk import word_tokenize

from ..utils import kerasmodel_io as kerasio
from ..utils import ModelNotTrainedException


class VarNNEmbeddedVecClassifier:
    def __init__(self,
                 wvmodel,
                 vecsize=300,
                 maxlen=15):
        self.wvmodel = wvmodel
        self.vecsize = vecsize
        self.maxlen = maxlen
        self.trained = False

    def convert_trainingdata_matrix(self, classdict):
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
                phrases.append(word_tokenize(shorttext))

        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)

        return classlabels, train_embedvec, indices


    def train(self, classdict, kerasmodel, nb_epoch=10):
        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix(classdict)

        # train the model
        kerasmodel.fit(train_embedvec, indices, nb_epoch=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

    def savemodel(self, nameprefix):
        if not self.trained:
            raise ModelNotTrainedException()
        kerasio.save_model(nameprefix, self.model)
        labelfile = open(nameprefix+'_classlabels.txt', 'w')
        labelfile.write('\n'.join(self.classlabels))
        labelfile.close()

    def loadmodel(self, nameprefix):
        self.model = kerasio.load_model(nameprefix)
        labelfile = open(nameprefix+'_classlabels.txt', 'r')
        self.classlabels = labelfile.readlines()
        labelfile.close()
        self.classlabels = map(lambda s: s.strip(), self.classlabels)
        self.trained = True

    def word_to_embedvec(self, word):
        return self.wvmodel[word] if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_matrix(self, shorttext):
        tokens = word_tokenize(shorttext)
        matrix = np.zeros((self.maxlen, self.vecsize))
        for i in range(min(self.maxlen, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix

    def score(self, shorttext):
        if not self.trained:
            raise ModelNotTrainedException()

        # retrieve vector
        matrix = np.array([self.shorttext_to_matrix(shorttext)])

        # classification using the neural network
        predictions = self.model.predict(matrix)

        # wrangle output result
        scoredict = {}
        for idx, classlabel in zip(range(len(self.classlabels)), self.classlabels):
            scoredict[classlabel] = predictions[0][idx]
        return scoredict