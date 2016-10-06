import numpy as np
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
from nltk import word_tokenize

from utils.classification_exceptions import ModelNotTrainedException
import utils.kerasmodel_io as kerasio

# ref: https://gist.github.com/entron/b9bc61a74e7cadeb1fec
# ref: http://cs231n.github.io/convolutional-networks/

class CNNEmbeddedVecClassifier:
    def __init__(self,
                 wvmodel,
                 classdict,
                 n_gram,
                 vecsize=300,
                 nb_filters=1200,
                 maxlen=15):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.n_gram = n_gram
        self.vecsize = vecsize
        self.nb_filters = nb_filters
        self.maxlen = maxlen
        self.trained = False

    def convert_trainingdata_matrix(self):
        classlabels = self.classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        # tokenize the words, and determine the word length
        phrases = []
        indices = []
        for label in classlabels:
            for shorttext in self.classdict[label]:
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


    def train(self):
        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix()

        # build the deep neural network model
        model = Sequential()
        model.add(Convolution1D(nb_filter=self.nb_filters,
                                filter_length=self.n_gram,
                                border_mode='valid',
                                activation='relu',
                                input_shape=(self.maxlen, self.vecsize)))
        model.add(MaxPooling1D(pool_length=self.maxlen-self.n_gram+1))
        model.add(Flatten())
        model.add(Dense(len(self.classlabels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_embedvec, indices)

        # flag switch
        self.model = model
        self.trained = True

    def savemodel(self, nameprefix):
        if not self.trained:
            raise ModelNotTrainedException()
        kerasio.save_model(nameprefix, self.model)

    def loadmodel(self, nameprefix):
        self.model = kerasio.load_model(nameprefix)
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