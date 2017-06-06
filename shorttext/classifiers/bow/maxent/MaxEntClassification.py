
from scipy.sparse import dok_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

from shorttext.utils import tokenize
from shorttext.utils import gensim_corpora as gc
from shorttext.utils import classification_exceptions as e

def logistic_framework(nb_inputs, nb_outputs, l2reg=0.01, bias_l2reg=0.01, optimizer='adam'):
    kmodel = Sequential()
    kmodel.add(Dense(units=nb_outputs,
                     activation='softmax',
                     input_shape=(1, nb_inputs),
                     kernel_regularizer=l2(l2reg),
                     bias_regularizer=l2(bias_l2reg))
               )
    kmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return kmodel

class MaxEntClassifier:
    def __init__(self, preprocessor=lambda s: s.lower()):
        self.preprocessor = preprocessor
        self.trained = False

    def shorttext_to_vec(self, shorttext):
        tokens = tokenize(self.preprocessor(shorttext))

        vec = dok_matrix((1, len(self.dictionary)))
        for token in tokens:
            vec[0, self.dictionary.token2id[token]] = 1.0

        return vec[0, :]

    def gensimcorpus_to_matrix(self, corpus):
        matrix = dok_matrix((len(corpus), len(self.dictionary)))
        for docid, doc in enumerate(corpus):
            for tokenid, count in doc:
                matrix[docid, tokenid] = count
        return matrix

    def index_classlabels(self):
        self.labels2idx = {label: idx for idx, label in enumerate(self.classlabels)}

    def convert_classdict_to_XY(self, classdict):
        nb_data = sum(map(lambda k: len(classdict[k]), classdict.keys()))
        X = dok_matrix((nb_data, len(self.dictionary)))
        y = dok_matrix((nb_data, len(self.labels2idx)))

        rowid = 0
        for label in classdict:
            if label in self.labels2idx.keys():
                for shorttext in classdict[label]:
                    X[rowid, :] = self.shorttext_to_vec(shorttext)
                    y[rowid, self.labels2idx[label]] = 1.
                    rowid += 1

        return X, y

    def train(self, classdict, nb_epoch=100, l2reg=0.01, bias_l2reg=0.01, optimizer='adam'):
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda s: tokenize(self.preprocessor(s)))
        self.index_classlabels()

        X, y = self.convert_classdict_to_XY(classdict)

        kmodel = logistic_framework(len(self.dictionary),
                                    len(self.classlabels),
                                    l2=l2reg,
                                    bias_l2reg=bias_l2reg,
                                    optimizer=optimizer)
        kmodel.fit(X, y, nb_epoch=nb_epoch)

        self.model = kmodel
        self.trained = True
