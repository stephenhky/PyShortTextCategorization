
from sklearn.svm import SVC
from loguru import logger

import shorttext
from shorttext.stack import StackedGeneralization, LogisticStackedGeneralization
from shorttext.smartload import smartload_compact_model
from shorttext.classifiers import TopicVectorSkLearnClassifier, TopicVectorCosineDistanceClassifier, MaxEntClassifier
from shorttext.generators import GensimTopicModeler, LDAModeler

import pytest


def training_stacking() -> tuple[MaxEntClassifier, GensimTopicModeler, TopicVectorSkLearnClassifier, StackedGeneralization]:
    # loading NIH Reports
    nihdict = shorttext.data.nihreports(sample_size=None)
    nihdict = {'NCCAM': nihdict['NCCAM'], 'NCATS': nihdict['NCATS']}

    # maxent
    maxent_classifier = MaxEntClassifier()
    maxent_classifier.train(nihdict, nb_epochs=100)
    maxent_classifier.save_compact_model('./bio_maxent.bin')

    # SVM + LDA
    topicmodeler = LDAModeler()
    topicmodeler.train(nihdict, 8)
    topicdisclassifier = TopicVectorCosineDistanceClassifier(topicmodeler)
    topicmodeler.save_compact_model('bio_lda.bin')
    svm_classifier = TopicVectorSkLearnClassifier(topicmodeler, SVC())
    svm_classifier.train(nihdict)
    svm_classifier.save_compact_model('bio_svm.bin')

    # logistic
    stacked_classifier = LogisticStackedGeneralization({
        'maxent': maxent_classifier,
        'svm': svm_classifier,
        'topiccosine': topicdisclassifier
    })
    stacked_classifier.train(nihdict, nb_epoch=300)
    stacked_classifier.save_compact_model('bio_logistics.bin')

    return maxent_classifier, topicmodeler, svm_classifier, stacked_classifier


def compare_two_dicts(dict1, dict2) -> None:
    assert len(dict1) == len(dict2)
    for classlabel in dict1:
        assert (classlabel in dict2)
        assert dict1[classlabel] == pytest.approx(dict2[classlabel], abs=1e-3)


def test_studies() -> None:
    # train
    maxent_classifier, topicmodeler, svm_classifier, stacked_classifier = training_stacking()
    topicdisclassifier = TopicVectorCosineDistanceClassifier(topicmodeler)

    # smartload
    maxent_classifier2 = smartload_compact_model('bio_maxent.bin', None)
    topicmodeler2 = smartload_compact_model('bio_lda.bin', None)
    topicdisclassifier2 = TopicVectorCosineDistanceClassifier(topicmodeler2)
    svm_classifier2 = smartload_compact_model('bio_svm.bin', None)
    stacked_classifier2 = LogisticStackedGeneralization({
        'maxent': maxent_classifier2,
        'svm': svm_classifier2,
        'topiccosine': topicdisclassifier2
    })
    stacked_classifier2.load_compact_model('bio_logistics.bin')

    # compare
    terms = ['stem cell', 'grant', 'system biology']
    for term in terms:
        logger.info(term)

        logger.info('maximum entropy')
        compare_two_dicts(maxent_classifier.score(term), maxent_classifier2.score(term))

        logger.info('LDA')
        compare_two_dicts(topicdisclassifier.score(term), topicdisclassifier2.score(term))

        logger.info('SVM')
        compare_two_dicts(svm_classifier.score(term), svm_classifier2.score(term))

        logger.info('combined')
        compare_two_dicts(stacked_classifier.score(term), stacked_classifier2.score(term))


def test_svm() -> None:
    # loading NIH Reports
    nihdict = shorttext.data.nihreports(sample_size=None)
    nihdict = {'NCCAM': nihdict['NCCAM'], 'NCATS': nihdict['NCATS']}

    # svm
    topicmodeler = LDAModeler()
    topicmodeler.train(nihdict, 16)
    svm_classifier = TopicVectorSkLearnClassifier(topicmodeler, SVC())
    svm_classifier.train(nihdict)

    logger.info('before saving...')
    logger.info('--'.join(svm_classifier.classlabels))
    svm_classifier.save_compact_model('bio_svm2.bin')
    logger.info('after saving...')
    logger.info('--'.join(svm_classifier.classlabels))

    # load
    svm_classifier2 = smartload_compact_model('bio_svm2.bin', None)
    logger.info('second classifier...')
    logger.info(','.join(svm_classifier2.classlabels))
    logger.info(','.join(svm_classifier2.topicmodeler.classlabels))

    # compare
    terms = ['stem cell', 'grant', 'system biology']
    for term in terms:
        logger.info(term)
        topicvec = svm_classifier.getvector(term)
        topicvec2 = svm_classifier2.getvector(term)

        logger.info(topicvec)
        logger.info(topicvec2)

        for idx, classlabel in enumerate(svm_classifier.classlabels):
            logger.info(f"{idx} {classlabel}")
            logger.info(svm_classifier.classifier.score([topicvec], [idx]))

        for idx, classlabel in enumerate(svm_classifier2.classlabels):
            logger.info(f"{idx} {classlabel}")
            logger.info(svm_classifier2.classifier.score([topicvec2], [idx]))

        logger.info({
            classlabel: svm_classifier.classifier.score([topicvec], [idx])
            for idx, classlabel in enumerate(svm_classifier.classlabels)
        })
        logger.info({
            classlabel: svm_classifier2.classifier.score([topicvec], [idx])
            for idx, classlabel in enumerate(svm_classifier2.classlabels)
        })

    for term in terms:
        logger.info(term)
        compare_two_dicts(svm_classifier.score(term), svm_classifier2.score(term))
