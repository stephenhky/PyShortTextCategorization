import os
import unittest
import urllib
import sys

import shorttext


# The "test_w2v_model.bin" in this directory is adapted from: https://raw.githubusercontent.com/chinmayapancholi13/shorttext_test_data/master/test_w2v_model

class TestVarNNEmbeddedVecClassifier(unittest.TestCase):
    def setUp(self):
        if not os.path.isfile("test_w2v_model.bin"):
            if sys.version_info[0]==2:
                urllib.urlretrieve(
                    "https://github.com/stephenhky/PyShortTextCategorization/blob/master/data/test_w2v_model.bin?raw=true",
                    "test_w2v_model.bin")
            else:
                urllib.request.urlretrieve(
                    "https://github.com/stephenhky/PyShortTextCategorization/blob/master/data/test_w2v_model.bin?raw=true",
                    "test_w2v_model.bin")
        self.w2v_model = shorttext.utils.load_word2vec_model("test_w2v_model.bin", binary=True)  # load word2vec model
        self.trainclass_dict = shorttext.data.subjectkeywords()  # load training data

    def tearDown(self):
        if os.path.isfile("test_w2v_model.bin"):
            os.remove('test_w2v_model.bin')

    def comparedict(self, dict1, dict2):
        self.assertTrue(len(dict1)==len(dict2))
        print(dict1, dict2)
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEqual(dict1[classlabel], dict2[classlabel], places=4)

    def testCNNWordEmbedWithoutGensim(self):
        # create keras model using `CNNWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model,
                                                                    nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testDoubleCNNWordEmbedWithoutGensim(self):
        # create keras model using `DoubleCNNWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model,
                                                                          nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testCLSTMWordEmbedWithoutGensim(self):
        # create keras model using `CLSTMWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model,
                                                                      nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testSumEmbed(self):
        classifier = shorttext.classifiers.SumEmbeddedVecClassifier(self.w2v_model)
        classdict = shorttext.data.subjectkeywords()
        classifier.train(classdict)

        # compute
        self.comparedict(classifier.score('linear algebra'),
                         {'mathematics': 0.9986082046096036,
                          'physics': 0.9976047404871671,
                          'theology': 0.9923434326310248})
        self.comparedict(classifier.score('learning'),
                         {'mathematics': 0.998968177605999,
                          'physics': 0.9995439648885027,
                          'theology': 0.9965552994894663})


# if __name__ == '__main__':
#     unittest.main()
