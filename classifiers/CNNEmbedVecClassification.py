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
        maxlen = max(map(len, phrases))

        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(phrases), maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(len(phrases[i])):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)

        return classlabels, maxlen, train_embedvec, indices


    def train(self):
        # convert classdict to training input vectors
        self.classlabels, self.maxlen, train_embedvec, indices = self.convert_trainingdata_matrix()

        # build the model
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
        # print train_embedvec.shape
        # print indices.shape
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

    def score(self, shorttext):
        if not self.trained:
            raise ModelNotTrainedException()

        tokens = word_tokenize(shorttext)
        matrix = np.zeros((1, len(tokens), self.vecsize))
        for i in range(len(tokens)):
            matrix[0, i] = self.word_to_embedvec(tokens[i])

        predictions = self.model.predict(matrix)
        scoredict = {}
        for idx, classlabel in zip(range(len(self.classlabels)), self.classlabels):
            scoredict[classlabel] = predictions[0][idx]
        return scoredict