
import pytest

from shorttext.metrics.dynprog.dldist import damerau_levenshtein
from shorttext.metrics.dynprog.lcp import longest_common_prefix
from shorttext.metrics.dynprog.jaccard import similarity as jaccard_similarity


def test_similarity():
    assert damerau_levenshtein('debug', 'deubg') == 1
    assert damerau_levenshtein('intrdependence', 'interdpeendencae') == 3
    assert longest_common_prefix('debug', 'debuag') == 4

def test_dldistance_transposition():
    assert damerau_levenshtein('independent', 'indeepndent') == 1
    assert damerau_levenshtein('providence', 'porvidecne') == 2

def test_dldistance_insertion():
    assert damerau_levenshtein('algorithm', 'algorithms') == 1
    assert damerau_levenshtein('algorithm', 'algoarithmm') == 2

def test_dldistance_deletion():
    assert damerau_levenshtein('algorithm', 'algoithm') == 1
    assert damerau_levenshtein('algorithm', 'algorith') == 1
    assert damerau_levenshtein('algorithm', 'algrihm') == 2

def test_dldistance_correct():
    assert damerau_levenshtein('python', 'python') == 0
    assert damerau_levenshtein('sosad', 'sosad') == 0

def test_dldistance_jaccard():
    assert jaccard_similarity('diver', 'driver') == pytest.approx(5/6)
