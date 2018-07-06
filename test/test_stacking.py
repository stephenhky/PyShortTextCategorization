
import unittest

import shorttext
from sklearn.svm import SVC

class TestStacking(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testNIH(self):
        # loading NIH Reports
        nihdict = shorttext.data.nihreports(sample_size=None)
        nihdict = {'NCCAM': nihdict['NCCAM'], 'NCATS': nihdict['NCATS']}

        # maxent
        maxent_classifier = shorttext.classifiers.MaxEntClassifier()
        maxent_classifier.train(nihdict, nb_epochs=100)

        # SVM + LDA
        topicmodeler = shorttext.generators.LDAModeler()
        topicmodeler.train(nihdict, 8)
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
        svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(topicmodeler, SVC())
        svm_classifier.train(nihdict)

        # logistic
        stacked_classifier = shorttext.stack.LogisticStackedGeneralization({'maxent': maxent_classifier,
                                                                            'svm': svm_classifier,
                                                                            'topiccosine': topicdisclassifier})
        stacked_classifier.train(nihdict)

        # test result
        self.assertEqual(sorted(stacked_classifier.score('single cell RNA sequencing').items(),
                                key = lambda item: item[1], reverse = True)[0][0],
                         'NCCAM')
        self.assertEqual(sorted(stacked_classifier.score('translational research').items(),
                                key = lambda item: item[1], reverse = True)[0][0],
                         'NCATS')
        self.assertEqual(sorted(stacked_classifier.score('stem cell').items(),
                                key = lambda item: item[1], reverse = True)[0][0],
                         'NCATS')


if __name__ == '__main__':
    unittest.main()

