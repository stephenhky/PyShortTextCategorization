
import unittest

import shorttext
from shorttext.stack import LogisticStackedGeneralization
from shorttext.smartload import smartload_compact_model
from sklearn.svm import SVC

class TestStacking(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def training(self):
        # loading NIH Reports
        nihdict = shorttext.data.nihreports(sample_size=None)
        nihdict = {'NCCAM': nihdict['NCCAM'], 'NCATS': nihdict['NCATS']}

        # maxent
        maxent_classifier = shorttext.classifiers.MaxEntClassifier()
        maxent_classifier.train(nihdict, nb_epochs=100)
        maxent_classifier.save_compact_model('./bio_maxent.bin')

        # SVM + LDA
        topicmodeler = shorttext.generators.LDAModeler()
        topicmodeler.train(nihdict, 8)
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
        topicmodeler.save_compact_model('./bio_lda.bin')
        svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(topicmodeler, SVC())
        svm_classifier.train(nihdict)
        svm_classifier.save_compact_model('./bio_svm.bin')

        # logistic
        stacked_classifier = LogisticStackedGeneralization({'maxent': maxent_classifier,
                                                            'svm': svm_classifier,
                                                            'topiccosine': topicdisclassifier})
        stacked_classifier.train(nihdict)
        stacked_classifier.save_compact_model('./bio_logistics.bin')

        return maxent_classifier, topicmodeler, svm_classifier, stacked_classifier

    def comparedict(self, dict1, dict2):
        self.assertTrue(len(dict1)==len(dict2))
        print(dict1, dict2)
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEquals(dict1[classlabel], dict2[classlabel], places=4)

    def testStudies(self):
        # train
        maxent_classifier, topicmodeler, svm_classifier, stacked_classifier = self.training()
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)

        # smartload
        maxent_classifier2 = smartload_compact_model('./bio_maxent.bin', None)
        topicmodeler2 = smartload_compact_model('./bio_lda.bin', None)
        topicdisclassifier2 = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler2)
        svm_classifier2 = smartload_compact_model('./bio_svm.bin', None)
        stacked_classifier2 = LogisticStackedGeneralization({'maxent': maxent_classifier2,
                                                             'svm': svm_classifier2,
                                                             'topiccosine': topicdisclassifier2})
        stacked_classifier2.load_compact_model('./bio_logistics.bin')

        # compare
        terms = ['stem cell', 'grant', 'system biology']
        for term in terms:
            print(term)
            print('maximum entropy')
            self.comparedict(maxent_classifier.score(term), maxent_classifier2.score(term))
            print('LDA')
            self.comparedict(topicdisclassifier.score(term), topicdisclassifier2.score(term))
            print('SVM')
            self.comparedict(svm_classifier.score(term), svm_classifier2.score(term))
            print('combined')
            self.comparedict(stacked_classifier.score(term), stacked_classifier2.score(term))


if __name__ == '__main__':
    unittest.main()

