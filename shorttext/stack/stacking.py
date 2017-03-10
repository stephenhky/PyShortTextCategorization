
import numpy as np

import utils.classification_exceptions as e

# abstract class
class StackedGeneralization:
    def __init__(self, intermediate_classifiers={}):
        self.classifiers = intermediate_classifiers
        self.classlabels = []

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

    def train(self, classdict):
        raise e.NotImplementedException()

    def score(self, shorttext):
        raise e.NotImplementedException()

class LogisticStackedGeneralization(StackedGeneralization):
    pass