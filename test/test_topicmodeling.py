
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pytest

import shorttext


def test_ldatopicmodel():
    # load data
    trainclassdict = shorttext.data.nihreports(sample_size=None)

    # train LDA model
    topicmodeler = shorttext.generators.LDAModeler()
    topicmodeler.train(trainclassdict, 128)

    # retrieve topic vectors
    topic_vector_1 = topicmodeler.retrieve_topicvec('stem cell research NIH cancer immunology')
    assert not np.any(np.isnan(topic_vector_1))
    assert np.linalg.norm(topic_vector_1) == pytest.approx(1.)

    topic_vector_2 = topicmodeler.retrieve_topicvec('bioinformatics')
    assert not np.any(np.isnan(topic_vector_2))
    assert np.linalg.norm(topic_vector_2) == pytest.approx(1.)

    topic_vector_3 = topicmodeler.retrieve_topicvec('linear algebra')
    assert not np.any(np.isnan(topic_vector_3))
    assert np.linalg.norm(topic_vector_3) == pytest.approx(1.)

    # test I/O
    topicmodeler.save_compact_model('nihlda128.bin')
    topicmodeler2 = shorttext.generators.load_gensimtopicmodel('nihlda128.bin')
    topic_vector_1a = topicmodeler2.retrieve_topicvec("stem cell research NIH cancer immunology")
    assert not np.any(np.isnan(topic_vector_1a))
    assert np.linalg.norm(topic_vector_1a) == pytest.approx(1.)
    # np.testing.assert_array_almost_equal(topic_vector_1a, topic_vector_1)  # do not check this; LDA models are stochastic

    # cosine similarity scorer
    cos_classifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
    score_dict = cos_classifier.score("stem cell research NIH cancer immunology")
    assert isinstance(score_dict, dict)
    assert len(score_dict) == len(trainclassdict)

    # scikit-learn classifier
    gaussian_nb_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(
        topicmodeler, GaussianNB()
    )
    gaussian_nb_classifier.train(trainclassdict)
    score_dict = gaussian_nb_classifier.score("stem cell research NIH cancer immunology")
    assert isinstance(score_dict, dict)


def test_autoencoder():
    # load data
    subdict = shorttext.data.subjectkeywords()

    # train the model
    autoencoder = shorttext.generators.AutoencodingTopicModeler()
    autoencoder.train(subdict, 8)

    # retrieve BOW vector
    bow_vector = autoencoder.retrieve_bow_vector("critical race")
    assert not np.any(np.isnan(bow_vector))
    assert np.all(bow_vector == 1 / np.sqrt(len(autoencoder.token2indices)))

    # retrieve topic vector
    topic_vector_1 = autoencoder.retrieve_topicvec("linear algebra")
    assert not np.any(np.isnan(topic_vector_1))
    assert np.linalg.norm(topic_vector_1) == pytest.approx(1.)
    np.testing.assert_array_almost_equal(autoencoder["linear algebra"], topic_vector_1)

    topic_vector_2 = autoencoder.retrieve_topicvec("path integral")
    assert not np.any(np.isnan(topic_vector_2))
    assert np.linalg.norm(topic_vector_2) == pytest.approx(1.)
    np.testing.assert_array_almost_equal(autoencoder["path integral"], topic_vector_2)

    topic_vector_3 = autoencoder.retrieve_topicvec("critical race")
    assert not np.any(np.isnan(topic_vector_3))
    assert np.linalg.norm(topic_vector_3) == pytest.approx(1.)
    np.testing.assert_array_almost_equal(autoencoder["critical race"], topic_vector_3)

    # cosine similarity scholar
    cos_classifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(autoencoder)
    score_dict = cos_classifier.score("stem cell research")
    assert isinstance(score_dict, dict)
    assert len(score_dict) == 3

    # scikit-learn classifier
    gaussian_nb_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(
        autoencoder, GaussianNB()
    )
    gaussian_nb_classifier.train(subdict)
    score_dict = gaussian_nb_classifier.score("path integral")
    assert isinstance(score_dict, dict)
