
import urllib
from pathlib import Path

from loguru import logger
import pytest

import shorttext


# download model
link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
filename = "test_w2v_model.bin"
if not Path(filename).exists():
    urllib.request.urlretrieve(link, filename)
w2v_model = shorttext.utils.load_word2vec_model(filename, binary=True)  # load word2vec model
trainclass_dict = shorttext.data.subjectkeywords()


def compare_two_dicts(dict1, dict2) -> None:
    assert len(dict1) == len(dict2)
    for classlabel in dict1:
        assert (classlabel in dict2)
        assert dict1[classlabel] == pytest.approx(dict2[classlabel], abs=1e-3)


def test_CNN_word_embed_without_gensim():
    logger.info("Testing CNN...")
    # create keras model using `CNNWordEmbed` class
    logger.info("\tKeras model")
    keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys())
    )

    # create and train classifier using keras model constructed above
    logger.info("\tTraining")
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=2)

    # compute classification score
    logger.info("\tTesting")
    score_vals = main_classifier.score('artificial intelligence')
    assert score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'] == pytest.approx(1.0)


def test_double_CNN_word_embed_ewithout_gensim():
    logger.info("Testing DoubleCNN...")
    # create keras model using `DoubleCNNWordEmbed` class
    logger.info("\tKeras model")
    keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys())
    )

    # create and train classifier using keras model constructed above
    logger.info("\tTraining")
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=2)

    # compute classification score
    logger.info("\tTesting")
    score_vals = main_classifier.score('artificial intelligence')
    assert score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'] == pytest.approx(1.0)


def test_CLSTM_word_embed_without_gensim():
    logger.info("Testing CLSTM...")
    # create keras model using `CLSTMWordEmbed` class
    logger.info("\tKeras model")
    keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(
        wvmodel=w2v_model,
        nb_labels=len(trainclass_dict.keys())
    )

    # create and train classifier using keras model constructed above
    logger.info("\tTraining")
    main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(w2v_model)
    main_classifier.train(trainclass_dict, keras_model, nb_epoch=2)

    # compute classification score
    logger.info("\tTesting")
    score_vals = main_classifier.score('artificial intelligence')
    assert score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'] ==pytest.approx(1.0)


def test_AA_sum_embed():
    logger.info("Testing SumEmbed")
    classifier = shorttext.classifiers.SumEmbeddedVecClassifier(w2v_model)
    classdict = shorttext.data.subjectkeywords()
    classifier.train(classdict)

    # compute
    compare_two_dicts(
        classifier.score('linear algebra'),
        {
            'mathematics': 0.9044698253778962,
            'physics': 0.7586816549044926,
            'theology': 0.1817602793151848
        }
    )
    compare_two_dicts(
        classifier.score('learning'),
        {
            'mathematics': 0.9037142562255835,
            'physics': 0.7588376500004107,
            'theology': 0.18039468994239538
        }
    )
    compare_two_dicts(
        classifier.score('eschatology'),
        {
            'mathematics': 0.3658578123294476,
            'physics': 0.5996711864493821,
            'theology': 0.9694560847986978
        }
    )
