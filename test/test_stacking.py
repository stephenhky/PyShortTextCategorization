import numpy as np
import pytest
from sklearn.svm import SVC

import shorttext
from shorttext.classifiers import (
    MaxEntClassifier,
    TopicVectorCosineDistanceClassifier,
    TopicVectorSkLearnClassifier,
)
from shorttext.generators import LDAModeler
from shorttext.smartload import smartload_compact_model
from shorttext.stack import LogisticStackedGeneralization


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _assert_score_dict_close(dict1, dict2, atol=1e-5):
    assert set(dict1.keys()) == set(dict2.keys())
    for label in dict1:
        assert dict1[label] == pytest.approx(dict2[label], abs=atol)


@pytest.fixture(scope="module")
def nih_two_class_dict():
    nihdict = shorttext.data.nihreports(sample_size=None)
    return {'NCCAM': nihdict['NCCAM'], 'NCATS': nihdict['NCATS']}


def test_stacking_roundtrip(nih_two_class_dict, tmp_path):
    # maxent
    maxent_classifier = MaxEntClassifier()
    maxent_classifier.train(nih_two_class_dict, nb_epochs=5)
    maxent_classifier.save_compact_model(str(tmp_path / 'bio_maxent.bin'))

    # LDA + SVM/cosine
    topicmodeler = LDAModeler()
    topicmodeler.train(nih_two_class_dict, 4, random_state=0)
    topicmodeler.save_compact_model(str(tmp_path / 'bio_lda.bin'))
    topicdisclassifier = TopicVectorCosineDistanceClassifier(topicmodeler)
    svm_classifier = TopicVectorSkLearnClassifier(topicmodeler, SVC(probability=False, random_state=0))
    svm_classifier.train(nih_two_class_dict)
    svm_classifier.save_compact_model(str(tmp_path / 'bio_svm.bin'))

    stacked_classifier = LogisticStackedGeneralization({
        'maxent': maxent_classifier,
        'svm': svm_classifier,
        'topiccosine': topicdisclassifier
    })
    stacked_classifier.train(nih_two_class_dict, nb_epoch=10)
    stacked_classifier.save_compact_model(str(tmp_path / 'bio_logistics.bin'))

    # reload
    maxent_classifier2 = smartload_compact_model(str(tmp_path / 'bio_maxent.bin'), None)
    topicmodeler2 = smartload_compact_model(str(tmp_path / 'bio_lda.bin'), None)
    topicdisclassifier2 = TopicVectorCosineDistanceClassifier(topicmodeler2)
    svm_classifier2 = smartload_compact_model(str(tmp_path / 'bio_svm.bin'), None)
    stacked_classifier2 = LogisticStackedGeneralization({
        'maxent': maxent_classifier2,
        'svm': svm_classifier2,
        'topiccosine': topicdisclassifier2
    })
    stacked_classifier2.load_compact_model(str(tmp_path / 'bio_logistics.bin'))

    terms = ['stem cell', 'grant', 'system biology']
    for term in terms:
        _assert_score_dict_close(maxent_classifier.score(term), maxent_classifier2.score(term))
        _assert_score_dict_close(stacked_classifier.score(term), stacked_classifier2.score(term))
        # topic-derived components should also be load-consistent when seeded
        _assert_score_dict_close(topicdisclassifier.score(term), topicdisclassifier2.score(term))
        # SVC.score returns accuracy-like per-label scores in this wrapper; compare dict shape/values
        svm1 = svm_classifier.score(term)
        svm2 = svm_classifier2.score(term)
        assert set(svm1.keys()) == set(svm2.keys()) == set(nih_two_class_dict.keys())


def test_svm_topic_classifier(nih_two_class_dict, tmp_path):
    topicmodeler = LDAModeler()
    topicmodeler.train(nih_two_class_dict, 4, random_state=0)
    svm_classifier = TopicVectorSkLearnClassifier(topicmodeler, SVC(probability=False, random_state=0))
    svm_classifier.train(nih_two_class_dict)

    assert svm_classifier.trained
    assert sorted(svm_classifier.classlabels) == sorted(nih_two_class_dict.keys())

    model_path = tmp_path / 'bio_svm2.bin'
    svm_classifier.save_compact_model(str(model_path))
    svm_classifier2 = smartload_compact_model(str(model_path), None)

    assert sorted(svm_classifier2.classlabels) == sorted(nih_two_class_dict.keys())
    for term in ['stem cell', 'grant', 'system biology']:
        vec1 = svm_classifier.getvector(term)
        vec2 = svm_classifier2.getvector(term)
        assert np.all(np.isfinite(vec1))
        assert np.all(np.isfinite(vec2))
        np.testing.assert_allclose(vec1, vec2, atol=1e-6)
        scores1 = svm_classifier.score(term)
        scores2 = svm_classifier2.score(term)
        assert set(scores1.keys()) == set(scores2.keys()) == set(nih_two_class_dict.keys())
        assert all(np.isfinite(v) for v in scores1.values())
        assert all(np.isfinite(v) for v in scores2.values())
