
from typing import Optional
from os import PathLike

import gensim

from .utils import standard_text_preprocessor_1
from .utils import compactmodel_io as cio
from .utils import classification_exceptions as e
from .classifiers import  load_varnnlibvec_classifier, load_sumword2vec_classifier
from .generators import load_autoencoder_topicmodel, load_gensimtopicmodel
from .generators import loadSeq2SeqWithKeras, loadCharBasedSeq2SeqGenerator
from .classifiers import load_autoencoder_topic_sklearnclassifier, load_gensim_topicvec_sklearnclassifier
from .classifiers import load_maxent_classifier


def smartload_compact_model(
        filename: str | PathLike,
        wvmodel: Optional[gensim.models.keyedvectors.KeyedVectors],
        preprocessor: Optional[callable] = None,
        vecsize: Optional[int] = None
):
    """ Load appropriate classifier or model from the binary model.

    The second parameter, `wvmodel`, can be set to `None` if no Word2Vec model is needed.

    :param filename: path of the compact model file
    :param wvmodel: Word2Vec model
    :param preprocessor: text preprocessor (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :param vecsize: length of embedded vectors in the model (Default: None, extracted directly from the word-embedding model)
    :return: appropriate classifier or model
    :raise: AlgorithmNotExistException
    :type filename: str
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type preprocessor: function
    :type vecsize: int
    """
    if preprocessor is None:
        preprocessor = standard_text_preprocessor_1()

    classifier_name = cio.get_model_classifier_name(filename)
    match classifier_name:
        case 'ldatopic' | 'lsitopic' | 'rptopic':
            return load_gensimtopicmodel(filename, preprocessor=preprocessor, compact=True)
        case 'kerasautoencoder':
            return load_autoencoder_topicmodel(filename, preprocessor=preprocessor, compact=True)
        case 'topic_sklearn':
            topicmodel = cio.get_model_config_field(filename, 'topicmodel')
            if topicmodel in ['ldatopic', 'lsitopic', 'rptopic']:
                return load_gensim_topicvec_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
            elif topicmodel in ['kerasautoencoder']:
                return load_autoencoder_topic_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
            else:
                raise e.AlgorithmNotExistException(topicmodel)
        case 'nnlibvec':
            return load_varnnlibvec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
        case 'sumvec':
            return load_sumword2vec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
        case 'maxent':
            return load_maxent_classifier(filename, compact=True)
        case 'kerasseq2seq':
            return loadSeq2SeqWithKeras(filename, compact=True)
        case 'charbases2s':
            return loadCharBasedSeq2SeqGenerator(filename, compact=True)
        case _:
            raise e.AlgorithmNotExistException(classifier_name)
