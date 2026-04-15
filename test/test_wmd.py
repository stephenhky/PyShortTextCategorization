
import urllib
from pathlib import Path

import pytest

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model


# download model
link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
filename = "test_w2v_model.bin"
if not Path(filename).exists():
    urllib.request.urlretrieve(link, filename)
w2v_model = load_word2vec_model(filename, binary=True)  # load word2vec model


def test_word_mover_distance_1():
    tokens1 = ['president', 'speaks']
    tokens2 = ['president', 'talks']
    known_answer = 0.19936788082122803
    wdistance = word_mover_distance(tokens1, tokens2, w2v_model)
    assert wdistance == pytest.approx(known_answer)


def test_word_mover_distance_2():
    tokens1 = ['fan', 'book']
    tokens2 = ['apple', 'orange']
    known_answer = 1.8019972145557404
    wdistance = word_mover_distance(tokens1, tokens2, w2v_model)
    assert wdistance == pytest.approx(known_answer)

