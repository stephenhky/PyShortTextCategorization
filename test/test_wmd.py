import os
import unittest
import urllib

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model


class TestWMD(unittest.TestCase):
    def setUp(self):
        print("Downloading word-embedding model....")
        link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
        filename = "test_w2v_model.bin"
        if not os.path.isfile("test_w2v_model.bin"):
            urllib.request.urlretrieve(link, filename)
        self.w2v_model = load_word2vec_model(filename, binary=True)  # load word2vec model

    def tearDown(self):
        print("Removing word-embedding model")
        if os.path.isfile("test_w2v_model.bin"):
            os.remove('test_w2v_model.bin')

    def calculate_wmd(self, tokens1, tokens2, answer):
        wdistance = word_mover_distance(tokens1, tokens2, self.w2v_model)
        self.assertAlmostEqual(wdistance, answer, delta=1e-3)

    def test_metrics(self):
        tokens1 = ['president', 'speaks']
        tokens2 = ['president', 'talks']
        known_answer = 0.19936788082122803
        self.calculate_wmd(tokens1, tokens2, known_answer)

        tokens1 = ['fan', 'book']
        tokens2 = ['apple', 'orange']
        known_answer = 1.8019972145557404
        self.calculate_wmd(tokens1, tokens2, known_answer)
