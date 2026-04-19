
import numpy as np
import pytest

from shorttext.utils.compute import cosine_similarity


def test_cosine_similarity_1():
    vec1 = np.array([0.3, 0.7])
    vec2 = np.array([-0.7, 0.3])
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.)


def test_cosine_similarity_2():
    vec1 = np.array([1., 1.])
    vec2 = np.array([2.5, 2.5])
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.)


def test_cosine_similarity_3():
    vec1 = np.array([3., 3.])
    vec2 = np.array([2., 0.])
    assert cosine_similarity(vec1, vec2) == pytest.approx(np.sqrt(0.5))
