import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import shorttext


pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def nih_trainclass_dict():
    return shorttext.data.nihreports(sample_size=None)


@pytest.fixture(scope="module")
def subject_trainclass_dict():
    return shorttext.data.subjectkeywords()


def _assert_valid_topic_vector(vec, expected_norm=1.0):
    assert vec is not None
    assert not np.any(np.isnan(vec))
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(expected_norm, abs=1e-6)


def _assert_score_dict(score_dict, labels):
    assert isinstance(score_dict, dict)
    assert set(score_dict.keys()) == set(labels)
    assert all(np.isfinite(v) for v in score_dict.values())


def test_ldatopicmodel(nih_trainclass_dict, tmp_path):
    topicmodeler = shorttext.generators.LDAModeler()
    topicmodeler.train(nih_trainclass_dict, 8, random_state=0)

    topic_vector_1 = topicmodeler.retrieve_topicvec('stem cell research NIH cancer immunology')
    _assert_valid_topic_vector(topic_vector_1)
    topic_vector_2 = topicmodeler.retrieve_topicvec('bioinformatics')
    _assert_valid_topic_vector(topic_vector_2)
    topic_vector_3 = topicmodeler.retrieve_topicvec('linear algebra')
    _assert_valid_topic_vector(topic_vector_3)

    # I/O round trip
    model_path = tmp_path / 'nihlda8.bin'
    topicmodeler.save_compact_model(str(model_path))
    topicmodeler2 = shorttext.generators.GensimTopicModeler.from_pretrained(str(model_path))
    topic_vector_1a = topicmodeler2.retrieve_topicvec("stem cell research NIH cancer immunology")
    _assert_valid_topic_vector(topic_vector_1a)
    np.testing.assert_allclose(topic_vector_1a, topic_vector_1, atol=1e-6)

    # cosine similarity scorer
    cos_classifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
    score_dict = cos_classifier.score("stem cell research NIH cancer immunology")
    _assert_score_dict(score_dict, nih_trainclass_dict.keys())

    # scikit-learn classifier
    lr_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(
        topicmodeler, LogisticRegression(random_state=0, max_iter=200)
    )
    lr_classifier.train(nih_trainclass_dict)
    score_dict = lr_classifier.score("stem cell research NIH cancer immunology")
    _assert_score_dict(score_dict, nih_trainclass_dict.keys())


def test_autoencoder(subject_trainclass_dict):
    autoencoder = shorttext.generators.AutoencodingTopicModeler()
    autoencoder.train(subject_trainclass_dict, 8, epochs=5, verbose=0, shuffle=False)

    bow_vector = autoencoder.retrieve_bow_vector("critical race")
    assert not np.any(np.isnan(bow_vector))
    assert bow_vector.shape[0] == len(autoencoder.token2indices)

    topic_vector_1 = autoencoder.retrieve_topicvec("linear algebra")
    _assert_valid_topic_vector(topic_vector_1)
    np.testing.assert_allclose(autoencoder["linear algebra"], topic_vector_1)

    topic_vector_2 = autoencoder.retrieve_topicvec("path integral")
    _assert_valid_topic_vector(topic_vector_2)
    np.testing.assert_allclose(autoencoder["path integral"], topic_vector_2)

    topic_vector_3 = autoencoder.retrieve_topicvec("critical race")
    _assert_valid_topic_vector(topic_vector_3)
    np.testing.assert_allclose(autoencoder["critical race"], topic_vector_3)

    cos_classifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(autoencoder)
    score_dict = cos_classifier.score("stem cell research")
    _assert_score_dict(score_dict, subject_trainclass_dict.keys())

    lr_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(
        autoencoder, LogisticRegression(random_state=0, max_iter=200)
    )
    lr_classifier.train(subject_trainclass_dict)
    score_dict = lr_classifier.score("path integral")
    _assert_score_dict(score_dict, subject_trainclass_dict.keys())
