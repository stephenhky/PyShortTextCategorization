import os
import unittest

import shorttext


class TestVarNNEmbeddedVecClassifier(unittest.TestCase):
    def setUp(self):
		if not os.path.isfile("test_w2v_model"):
			os.system("wget https://raw.githubusercontent.com/chinmayapancholi13/shorttext_test_data/master/test_w2v_model")  # download w2v model 

		self.w2v_model = shorttext.utils.load_word2vec_model("test_w2v_model", binary=False)  # load word2vec model
		self.trainclass_dict = shorttext.data.subjectkeywords()  # load training data

    def tearDown(self):
		if os.path.isfile("test_w2v_model"):
			os.remove("test_w2v_model")  # delete downloaded w2v model

    def testCNNWordEmbedWithoutGensim(self):
  		# create keras model using `CNNWordEmbed` class
  		keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=False)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=False, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testCNNWordEmbedWithGensim(self):
		# create keras model using `CNNWordEmbed` class
		keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=True)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=True, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testDoubleCNNWordEmbedWithoutGensim(self):
  		# create keras model using `DoubleCNNWordEmbed` class
  		keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=False)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=False, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testDoubleCNNWordEmbedWithGensim(self):
		# create keras model using `DoubleCNNWordEmbed` class
		keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=True)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=True, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testCLSTMWordEmbedWithoutGensim(self):
  		# create keras model using `CLSTMWordEmbed` class
  		keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=False)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=False, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def testCLSTMWordEmbedWithGensim(self):
  		# create keras model using `CLSTMWordEmbed` class
  		keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model, nb_labels=len(self.trainclass_dict.keys()), vecsize=None, with_gensim=True)

    	# create and train classifier using keras model constructed above
		main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model, with_gensim=True, vecsize=None)
		main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

		# compute classification score
		score_vals = main_classifier.score('artificial intelligence')
		self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

if __name__ == '__main__':
    unittest.main()
