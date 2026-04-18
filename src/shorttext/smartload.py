
from typing import Optional
from os import PathLike

import gensim

from .utils import standard_text_preprocessor_1
from .utils import compactmodel_io as cio
from .utils import classification_exceptions as e
from .classifiers import  load_varnnlibvec_classifier, load_sumword2vec_classifier
from .generators import load_autoencoder_topicmodel, load_gensimtopicmodel
from .generators import load_seq2seq_model, loadCharBasedSeq2SeqGenerator
from .classifiers import load_autoencoder_topic_sklearnclassifier, load_gensim_topicvec_sklearnclassifier
from .classifiers import load_maxent_classifier
from .utils.dtm import load_numpy_documentmatrixmatrix


def smartload_compact_model(
        filename: str | PathLike,
        wvmodel: Optional[gensim.models.keyedvectors.KeyedVectors],
        preprocessor: Optional[callable] = None,
        vecsize: Optional[int] = None
):
    """Load a classifier or model from a compact file.

    Automatically detects the model type and loads the appropriate classifier.
    Set wvmodel to None if no word embedding model is needed.

    Args:
        filename: Path to the compact model file.
        wvmodel: Word embedding model. Can be None for non-embedding models.
        preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
        vecsize: Vector size. Default: None (extracted from model).

    Returns:
        Appropriate classifier or model instance.

    Raises:
        AlgorithmNotExistException: If model type is unknown.
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
            return load_seq2seq_model(filename, compact=True)
        case 'charbases2s':
            return loadCharBasedSeq2SeqGenerator(filename, compact=True)
        case "npdtm":
            return load_numpy_documentmatrixmatrix(filename)
        case _:
            raise e.AlgorithmNotExistException(classifier_name)
