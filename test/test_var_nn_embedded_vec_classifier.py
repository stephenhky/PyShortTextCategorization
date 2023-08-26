import os
import unittest
import urllib

import shorttext


class TestVarNNEmbeddedVecClassifier(unittest.TestCase):
    def setUp(self):
        print("Downloading word-embedding model....")
        link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
        filename = "test_w2v_model.bin"
        if not os.path.isfile("test_w2v_model.bin"):
            urllib.request.urlretrieve(link, filename)
        self.w2v_model = shorttext.utils.load_word2vec_model(filename, binary=True)  # load word2vec model
        self.trainclass_dict = shorttext.data.subjectkeywords()  # load training data

    def tearDown(self):
        print("Removing word-embedding model")
        if os.path.isfile("test_w2v_model.bin"):
            os.remove('test_w2v_model.bin')

    def comparedict(self, dict1, dict2):
        self.assertTrue(len(dict1)==len(dict2))
        print(dict1, dict2)
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEqual(dict1[classlabel], dict2[classlabel], places=4)

    def testCNNWordEmbedWithoutGensim(self):
        print("Testing CNN...")
        # create keras model using `CNNWordEmbed` class
        print("\tKeras model")
        keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model,
                                                                    nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        print("\tTraining")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        print("\tTesting")
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testDoubleCNNWordEmbedWithoutGensim(self):
        print("Testing DoubleCNN...")
        # create keras model using `DoubleCNNWordEmbed` class
        print("\tKeras model")
        keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model,
                                                                          nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        print("\tTraining")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        print("\tTesting")
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testCLSTMWordEmbedWithoutGensim(self):
        print("Testing CLSTM...")
        # create keras model using `CLSTMWordEmbed` class
        print("\tKeras model")
        keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model,
                                                                      nb_labels=len(self.trainclass_dict.keys()))

        # create and train classifier using keras model constructed above
        print("\tTraining")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        print("\tTesting")
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testAASumEmbed(self):
        print("Testing SumEmbed")
        classifier = shorttext.classifiers.SumEmbeddedVecClassifier(self.w2v_model)
        classdict = shorttext.data.subjectkeywords()
        classifier.train(classdict)

        # compute
        self.comparedict(classifier.score('linear algebra'),
                         {'mathematics': 0.9044698253778962,
                          'physics': 0.7586816549044926,
                          'theology': 0.1817602793151848})
        self.comparedict(classifier.score('learning'),
                         {'mathematics': 0.9037142562255835,
                          'physics': 0.7588376500004107,
                          'theology': 0.18039468994239538})
        self.comparedict(classifier.score('eschatology'),
                         {'mathematics': 0.3658578123294476,
                          'physics': 0.5996711864493821,
                          'theology': 0.9694560847986978})


if __name__ == '__main__':
    unittest.main()
