import math

import pytest

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model


@pytest.fixture(scope="module")
def w2v_model(test_w2v_model_path):
    return load_word2vec_model(str(test_w2v_model_path), binary=True)


def test_word_mover_distance_closer_pairs(w2v_model):
    tokens1 = ['president', 'speaks']
    tokens2 = ['president', 'talks']
    close_dist = word_mover_distance(tokens1, tokens2, w2v_model)
    far_dist = word_mover_distance(['fan', 'book'], ['apple', 'orange'], w2v_model)
    assert close_dist >= 0.0
    assert far_dist >= 0.0
    assert close_dist < far_dist


def test_word_mover_distance_identity(w2v_model):
    tokens = ['president', 'speaks']
    assert word_mover_distance(tokens, tokens, w2v_model) == pytest.approx(0.0, abs=1e-8)
