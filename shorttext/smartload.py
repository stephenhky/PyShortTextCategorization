
from .utils import standard_text_preprocessor_1
from .utils import compactmodel_io as cio
from .utils import classification_exceptions as e
from .classifiers import  load_varnnlibvec_classifier, load_sumword2vec_classifier
from .generators import load_autoencoder_topicmodel, load_gensimtopicmodel
from .classifiers import load_autoencoder_topic_sklearnclassifier, load_gensim_topicvec_sklearnclassifier
# from .classifiers import load_maxent_classifier


def smartload_compact_model(filename, wvmodel, preprocessor=standard_text_preprocessor_1()):
    """ Load appropriate classifier or model from the binary model.

    The second parameter, `wvmodel`, can be set to `None` if no Word2Vec model is needed.

    :param filename: path of the compact model file
    :param wvmodel: Word2Vec model
    :param preprocessor: text preprocessor (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :return: appropriate classifier or model
    :raise: AlgorithmNotExistException
    :type filename: str
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type preprocessor: function
    """
    classifier_name = cio.get_model_classifier_name(filename)
    if classifier_name in ['ldatopic', 'lsitopic', 'rptopic']:
        return load_gensimtopicmodel(filename, preprocessor=preprocessor, compact=True)
    elif classifier_name in ['kerasautoencoder']:
        return load_autoencoder_topicmodel(filename, preprocessor=preprocessor, compact=True)
    elif classifier_name in ['topic_sklearn']:
        topicmodel = cio.get_model_config_field(filename, 'topicmodel')
        if topicmodel in ['ldatopic', 'lsitopic', 'rptopic']:
            return load_gensim_topicvec_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        elif topicmodel in ['kerasautoencoder']:
            return load_autoencoder_topic_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        else:
            raise e.AlgorithmNotExistException(topicmodel)
    elif classifier_name in ['nnlibvec']:
        return load_varnnlibvec_classifier(wvmodel, filename, compact=True)
    elif classifier_name in ['sumvec']:
        return load_sumword2vec_classifier(wvmodel, filename, compact=True)
    # elif classifier_name in ['maxent']:
    #     return load_maxent_classifier(filename, compact=True)
    else:
        raise e.AlgorithmNotExistException(classifier_name)