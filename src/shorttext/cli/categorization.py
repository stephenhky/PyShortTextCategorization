
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Optional

from loguru import logger
import click

from ..utils.compactmodel_io import get_model_classifier_name
from ..utils.classification_exceptions import AlgorithmNotExistException, WordEmbeddingModelNotExistException
from ..utils import load_word2vec_model, load_fasttext_model, load_poincare_model
from ..smartload import smartload_compact_model
from ..classifiers import TopicVectorCosineDistanceClassifier


# configs
allowed_classifiers = [
    'ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder',
    'topic_sklearn', 'nnlibvec', 'sumvec', 'maxent'
]
needembedded_classifiers = ['nnlibvec', 'sumvec']
topicmodels = ['ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder']


# lazy functions for loading word embedding model
load_word2vec_nonbinary_model = partial(load_word2vec_model, binary=False)
load_poincare_binary_model = partial(load_poincare_model, binary=True)

typedict = {
    'word2vec': load_word2vec_model,
    'word2vec_nonbinary': load_word2vec_nonbinary_model,
    'fasttext': load_fasttext_model,
    'poincare': load_poincare_model,
    'poincare_binary': load_poincare_binary_model
}


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.option("--wv", default=None, type=click.Path(exists=False),
              help='Path of the pre-trained Word2Vec model, if needed.')
@click.option("--vecsize", default=300, type=int, help='Vector dimensions. (Default: 300)')
@click.option("--topn", default=10, type=int, help='Number of top results to show.')
@click.option("--inputtext", default=None, type=str,
              help='Single input text for classification. If omitted, will enter console mode.')
@click.option("--type", default="word2vec", type=click.Choice(list(typedict.keys())),
              help='Type of word-embedding model (default: word2vec)')
def shorttext_categorize(
        model_filepath: Path,
        wv: Optional[Path],
        vecsize: int,
        topn: int,
        inputtext: Optional[str],
        type: str
) -> None:
    """
    Perform prediction on short text with a given trained model.

    \b
    MODEL_FILEPATH     Path of the trained (compact) model.
    """
    # path normalization
    model_filepath = Path(model_filepath)

    # check if the model file is given
    if not model_filepath.exists():
        raise IOError(f'Model file "{model_filepath.as_posix()}" not found!')
    
    # get the name of the classifier
    logger.info('Retrieving classifier name...')
    classifier_name = get_model_classifier_name(model_filepath)

    if classifier_name not in allowed_classifiers:
        raise AlgorithmNotExistException(classifier_name)

    # load the Word2Vec model if necessary
    wvmodel = None
    if classifier_name in needembedded_classifiers:
        if wv is None:
            raise ValueError("You need to specify the path of an embedding model using --wv")
        wv = Path(wv)
        # check if the word embedding model is available
        if not wv.exists():
            raise WordEmbeddingModelNotExistException(wv)
        # if there, load it
        logger.info(f'Loading word-embedding model from {wv.as_posix()}...')
        wvmodel = typedict[type](wv)

    # load the classifier
    logger.info('Initializing the classifier...')
    if classifier_name in topicmodels:
        topicmodel = smartload_compact_model(model_filepath, wvmodel, vecsize=vecsize)
        classifier = TopicVectorCosineDistanceClassifier(topicmodel)
    else:
        classifier = smartload_compact_model(model_filepath, wvmodel, vecsize=vecsize)

    # predict single input or run in console mode
    if inputtext is not None:
        if len(inputtext.strip()) == 0:
            print('No input text provided.')
            return
        scoredict = classifier.score(inputtext)
        for label, score in sorted(scoredict.items(), key=itemgetter(1), reverse=True)[:topn]:
            print(f'{label} : {score:.4f}')
    else:
        # Console 
        print('Enter text to classify (empty input to quit):')
        while True:
            shorttext = input('text> ').strip()
            if not shorttext:
                break
            scoredict = classifier.score(shorttext)
            for label, score in sorted(scoredict.items(), key=itemgetter(1), reverse=True)[:topn]:
                print(f'{label} : {score:.4f}')
        print('Done.')
