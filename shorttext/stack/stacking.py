
import numpy as np
from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.regularizers import l2

import utils.classification_exceptions as e

# abstract class
class StackedGeneralization:
    def __init__(self, intermediate_classifiers={}):
        self.classifiers = intermediate_classifiers
        self.classlabels = []
        self.trained = False

    def register_classifiers(self):
        self.classifier2idx = {}
        self.idx2classifier = {}
        for idx, key in enumerate(self.classifiers.keys()):
            self.classifier2idx[key] = idx
            self.idx2classifier[idx] = key

    def register_classlabels(self, labels):
        self.classlabels = list(labels)
        self.labels2idx = {}
        for idx, classlabel in enumerate(self.classlabels):
            self.labels2idx[classlabel] = idx

    def add_classifier(self, name, classifier):
        self.classifiers[name] = classifier
        self.register_classifiers()

    def delete_classifier(self, name):
        del self.classifiers[name]
        self.register_classifiers()

    def translate_shorttext_intfeature_matrix(self, shorttext):
        feature_matrix = np.zeros((len(self.classifier2idx), len(self.labels2idx)))
        for key in self.classifier2idx:
            scoredict = self.classifiers[key].score(shorttext)
            for label in scoredict:
                feature_matrix[self.classifier2idx[key], self.labels2idx[label]] = scoredict[label]
        return feature_matrix

    def convert_label_to_buckets(self, label):
        buckets = np.zeros(len(self.labels2idx), dtype=np.int)
        buckets[self.labels2idx[label]] = 1
        return buckets

    def convert_traindata_matrix(self, classdict, tobucket=True):
        for label in classdict:
            y = self.convert_label_to_buckets(label) if tobucket else self.labels2idx[label]
            for shorttext in classdict[label]:
                X = self.translate_shorttext_intfeature_matrix(shorttext)
                yield X, y

    def train(self, classdict):
        raise e.NotImplementedException()

    def score(self, shorttext):
        raise e.NotImplementedException()

class LogisticStackedGeneralization(StackedGeneralization):
    def train(self, classdict, optimizer='adam', l2reg=0.01, nb_epoch=100):
        kmodel = Sequential()
        kmodel.add(Reshape((len(self.classifier2idx) * len(self.labels2idx),),
                           input_shape=(len(self.classifier2idx), len(self.labels2idx))))
        kmodel.add(Dense(output_dim=len(classdict),
                         activation='sigmoid',
                         W_regularizer=l2(l2reg)))
        kmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)

        Xy = [(xone, yone) for xone, yone in self.convert_traindata_matrix(classdict, tobucket=True)]
        X = np.array(map(lambda item: item[0], Xy))
        y = np.array(map(lambda item: item[1], Xy))

        print X.shape, y.shape

        kmodel.fit(X, y, nb_epoch=nb_epoch)

        self.model = kmodel
        self.trained = True

    def score(self, shorttext):
        if not self.trained:
            raise e.ModelNotTrainedException()

        input_matrix = self.translate_shorttext_intfeature_matrix(shorttext)
        prediction = self.model.predict(np.array([input_matrix]))

        scoredict = {}
        for idx, label in enumerate(self.classlabels):
            scoredict[label] = prediction[0][idx]

        return scoredict