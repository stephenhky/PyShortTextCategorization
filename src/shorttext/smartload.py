
from typing import Optional
from os import PathLike

import gensim

from .utils import standard_text_preprocessor_1
from .utils import compactmodel_io as cio
from .utils import classification_exceptions as e
from .classifiers import VarNNEmbeddedVecClassifier, SumEmbeddedVecClassifier
from .generators import GensimTopicModeler
from .generators.bow.AutoEncodingTopicModeling import AutoencodingTopicModeler
from .generators import CharBasedSeq2SeqGenerator, Seq2SeqWithKeras
from .classifiers import TopicVectorSkLearnClassifier
from .classifiers.bow.maxent.MaxEntClassification import MaxEntClassifier
from .utils.dtm import NumpyDocumentTermMatrix


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
            return GensimTopicModeler.from_pretrained(filename, preprocessor=preprocessor, compact=True)
        case 'kerasautoencoder':
            return AutoencodingTopicModeler.from_pretrained(filename, preprocessor=preprocessor, compact=True)
        case 'topic_sklearn':
            topicmodel = cio.get_model_config_field(filename, 'topicmodel')
            if topicmodel in ['ldatopic', 'lsitopic', 'rptopic']:
                return TopicVectorSkLearnClassifier.from_pretrained_gensimtopic_sklearnclassifier(
                    filename, preprocessor=preprocessor, compact=True
                )
            elif topicmodel in ['kerasautoencoder']:
                return TopicVectorSkLearnClassifier.from_pretrained_autoencoder_sklearnclassifier(
                    filename, preprocessor=preprocessor, compact=True
                )
            else:
                raise e.AlgorithmNotExistException(topicmodel)
        case 'nnlibvec':
            return VarNNEmbeddedVecClassifier.from_pretrained(wvmodel, filename, compact=True, vecsize=vecsize)
        case 'sumvec':
            return SumEmbeddedVecClassifier.from_pretrained(wvmodel, filename, compact=True, vecsize=vecsize)
        case 'maxent':
            return MaxEntClassifier.from_pretrained(filename, compact=True)
        case 'kerasseq2seq':
            return Seq2SeqWithKeras.from_pretrained(filename, compact=True)
        case 'charbases2s':
            return CharBasedSeq2SeqGenerator.from_pretrained(filename, compact=True)
        case "npdtm":
            return NumpyDocumentTermMatrix.from_npdict_file(filename)
        case _:
            raise e.AlgorithmNotExistException(classifier_name)
