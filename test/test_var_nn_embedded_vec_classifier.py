import math

import numpy as np
import pytest

import shorttext


pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def w2v_model(test_w2v_model_path):
    return shorttext.utils.load_word2vec_model(str(test_w2v_model_path), binary=True)


@pytest.fixture(scope="module")
def trainclass_dict():
    return shorttext.data.subjectkeywords()


def _assert_valid_score_dict(score_vals, expected_labels):
    assert set(score_vals.keys()) == set(expected_labels)
    values = np.array(list(score_vals.values()), dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all((values >= -1e-8) & (values <= 1.0 + 1e-8))
    assert float(np.sum(values)) == pytest.approx(1.0, abs=1e-5)


def test_cnn_word_embed_sumvec(w2v_model, trainclass_dict):
    keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys()),
        vecsize=w2v_model.vector_size
    )
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=1)
    score_vals = main_classifier.score('artificial intelligence')
    _assert_valid_score_dict(score_vals, trainclass_dict.keys())


def test_double_cnn_word_embed_sumvec(w2v_model, trainclass_dict):
    keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys()),
        vecsize=w2v_model.vector_size
    )
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=1)
    score_vals = main_classifier.score('artificial intelligence')
    _assert_valid_score_dict(score_vals, trainclass_dict.keys())


def test_clstm_word_embed_sumvec(w2v_model, trainclass_dict):
    keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys()),
        vecsize=w2v_model.vector_size
    )
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=1)
    score_vals = main_classifier.score('artificial intelligence')
    _assert_valid_score_dict(score_vals, trainclass_dict.keys())


def test_sum_embed_classifier_deterministic(w2v_model):
    classifier = shorttext.classifiers.SumEmbeddedVecClassifier(w2v_model)
    classdict = shorttext.data.subjectkeywords()
    classifier.train(classdict)

    score_vals = classifier.score('linear algebra')
    assert set(score_vals.keys()) == set(classdict.keys())
    values = np.array(list(score_vals.values()), dtype=float)
    assert np.all(np.isfinite(values))
    # same call under seeded environment should be stable
    score_vals2 = classifier.score('linear algebra')
    for label in score_vals:
        assert score_vals[label] == pytest.approx(score_vals2[label], abs=1e-12)

    # with this tiny fixture, math/physics-related terms should score higher on those classes than theology
    assert score_vals['mathematics'] > score_vals['theology']
