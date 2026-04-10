
from typing import Optional, Literal, Any

import gensim
import numpy as np
import numpy.typing as npt
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, LsiModel, RpModel
from gensim.similarities import MatrixSimilarity
import orjson

from ...utils import classification_exceptions as e
from ...utils.compactmodel_io import CompactIOMachine, get_model_classifier_name
from ...utils import gensim_corpora as gc
from .LatentTopicModeling import LatentTopicModeler


gensim_topic_model_dict = {'lda': LdaModel, 'lsi': LsiModel, 'rp': RpModel}


class GensimTopicModeler(LatentTopicModeler):
    """
    This class facilitates the creation of topic models (options: LDA (latent Dirichlet Allocation),
    LSI (latent semantic indexing), and Random Projections (RP))
    with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.

    No compact model I/O available for this class. Refer to
    :class:`LDAModeler` and :class:`LSIModeler`.

    This class extends :class:`LatentTopicModeler`.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            algorithm: Literal["lda", "lsi", "rp"] = "lda",
            toweigh: bool = True,
            normalize: bool = True
    ):
        """ Initialize the topic modeler.

        :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
        :param algorithm: algorithm for topic modeling. Options: lda, lsi, rp. (Default: lda)
        :param toweigh: whether to weigh the words using tf-idf. (Default: True)
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :type preprocessor: function
        :type algorithm: str
        :type toweigh: bool
        """
        LatentTopicModeler.__init__(
            self, preprocessor=preprocessor, tokenizer=tokenizer, normalize=normalize
        )
        self.algorithm = algorithm
        self.toweigh = toweigh

    def generate_corpus(self, classdict: dict[str, list[str]]) -> None:
        """ Calculate the gensim dictionary and corpus, and extract the class labels
        from the training data. Called by :func:`~train`.

        :param classdict: training data
        :return: None
        :type classdict: dict
        """
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(
            classdict,
            preprocess_and_tokenize=lambda sent: self.tokenize_func(self.preprocess_func(sent))
        )

    def train(self, classdict: dict[str, list[str]], nb_topics: int, *args, **kwargs) -> None:
        """ Train the topic modeler.

        :param classdict: training data
        :param nb_topics: number of latent topics
        :param args: arguments to pass to the `train` method for gensim topic models
        :param kwargs: arguments to pass to the `train` method for gensim topic models
        :return: None
        :type classdict: dict
        :type nb_topics: int
        """
        self.nb_topics = nb_topics
        self.generate_corpus(classdict)
        if self.toweigh:
            self.tfidf = TfidfModel(self.corpus)
            normcorpus = self.tfidf[self.corpus]
        else:
            self.tfidf = None
            normcorpus = self.corpus

        self.topicmodel = gensim_topic_model_dict[self.algorithm](
            normcorpus, num_topics=self.nb_topics, *args, **kwargs
        )
        self.matsim = MatrixSimilarity(self.topicmodel[normcorpus])

        # change the flag
        self.trained = True

    def update(self, additional_classdict: dict[str, list[str]]) -> None:
        """ Update the model with additional data.
        
        It updates the topic model with additional data.
        
        Warning: It does not allow adding class labels, and new words.
        The dictionary is not changed. Therefore, such an update will alter the
        topic model only. It affects the topic vector representation. While the corpus
        is changed, the words pumped into calculating the similarity matrix is not changed.
        
        Therefore, this function means for a fast update.
        But if you want a comprehensive model, it is recommended to retrain.
        
        :param additional_classdict: additional training data
        :return: None
        :type additional_classdict: dict
        """
        # cannot use this way, as we want to update the corpus with existing words
        self.corpus, newcorpus = gc.update_corpus_labels(
            self.dictionary,
            self.corpus,
            additional_classdict,
            preprocess_and_tokenize=lambda sent: self.tokenize_func(self.preprocess_func(sent))
        )
        self.topicmodel.update(newcorpus)

    def retrieve_bow(self, shorttext: str) -> list[tuple[int, int]]:
        """ Calculate the gensim bag-of-words representation of the given short text.

        :param shorttext: text to be represented
        :return: corpus representation of the text
        :type shorttext: str
        :rtype: list
        """
        return self.dictionary.doc2bow(self.tokenize_func(self.preprocess_func(shorttext)))

    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        """ Calculate the vector representation of the bag-of-words in terms of numpy.ndarray.

        :param shorttext: short text
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :return: vector represtation of the text
        :type shorttext: str
        :type normalize: bool
        :rtype: numpy.ndarray
        """
        bow = self.retrieve_bow(shorttext)
        vec = np.zeros(len(self.dictionary))
        if len(bow) > 0:
            for id, val in bow:
                vec[id] = val
        else:
            vec = 1.
        if self.normalize:
            vec /= np.linalg.norm(vec)
        return vec

    def retrieve_corpus_topicdist(self, shorttext: str) -> list[tuple[int, int | float]]:
        """ Calculate the topic vector representation of the short text, in the corpus form.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: text to be represented
        :return: topic vector in the corpus form
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: list
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        bow = self.retrieve_bow(shorttext)
        return self.topicmodel[self.tfidf[bow] if self.toweigh else bow]

    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
        """ Calculate the topic vector representation of the short text.

        This function calls :func:`~retrieve_corpus_topicdist`.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: text to be represented
        :return: topic vector
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        topicdist = self.retrieve_corpus_topicdist(shorttext)
        topicvec = np.zeros(self.nb_topics)
        for topicid, frac in topicdist:
            topicvec[topicid] = frac
        if self.normalize:
            topicvec /= np.linalg.norm(topicvec)
        return topicvec

    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
        """ Calculate the score, which is the cosine similarity with the topic vector of the model,
        of the short text against each class labels.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: short text
        :return: dictionary of scores of the text to all classes
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: dict
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        simdict = {}
        similarities = self.matsim[self.retrieve_corpus_topicdist(shorttext)]
        for label, similarity in zip(self.classlabels, similarities):
            simdict[label] = float(similarity)
        return simdict

    def loadmodel(self, nameprefix: str) -> None:
        """ Load the topic model with the given prefix of the file paths.

        Given the prefix of the file paths, load the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        :param nameprefix: prefix of the file paths
        :return: None
        :type nameprefix: str
        """
        # load the JSON file (parameters)
        parameters = orjson.loads(open(nameprefix+'.json', 'rb').read())
        self.nb_topics = parameters['nb_topics']
        self.toweigh = parameters['toweigh']
        self.algorithm = parameters['algorithm']
        self.classlabels = parameters['classlabels']

        # load the dictionary
        self.dictionary = Dictionary.load(nameprefix+'.gensimdict')

        # load the topic model
        self.topicmodel = gensim_topic_model_dict[self.algorithm].load(nameprefix + '.gensimmodel')

        # load the similarity matrix
        self.matsim = MatrixSimilarity.load(nameprefix+'.gensimmat')

        # load the tf-idf modek
        if self.toweigh:
            self.tfidf = TfidfModel.load(nameprefix+'.gensimtfidf')

        # flag
        self.trained = True

    def savemodel(self, nameprefix: str) -> None:
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param nameprefix: prefix of the file paths
        :return: None
        :raise: ModelNotTrainedException
        :type nameprefix: str
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        parameters = {}
        parameters['nb_topics'] = self.nb_topics
        parameters['toweigh'] = self.toweigh
        parameters['algorithm'] = self.algorithm
        parameters['classlabels'] = self.classlabels
        open(nameprefix+".json", "wb").write(orjson.dumps(parameters))

        self.dictionary.save(nameprefix+'.gensimdict')
        self.topicmodel.save(nameprefix+'.gensimmodel')
        self.matsim.save(nameprefix+'.gensimmat')
        if self.toweigh:
            self.tfidf.save(nameprefix+'.gensimtfidf')


lda_suffices =  ['.json', '.gensimdict', '.gensimmodel.state',
                   '.gensimtfidf', '.gensimmodel', '.gensimmat']
if gensim.__version__ >= '1.0.0':
    lda_suffices += ['.gensimmodel.expElogbeta.npy', '.gensimmodel.id2word']


class LDAModeler(GensimTopicModeler, CompactIOMachine):
    """
    This class facilitates the creation of LDA (latent Dirichlet Allocation) topic models,
    with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.

    This class extends :class:`GensimTopicModeler`.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            toweigh: bool = True,
            normalize: bool = True
    ):
        GensimTopicModeler.__init__(
            self,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            algorithm="lda",
            toweigh=toweigh,
            normalize=normalize
        )
        CompactIOMachine.__init__(
            self, {'classifier': 'ldatopic'}, 'ldatopic', lda_suffices
        )

    def getinfo(self) -> dict[str, Any]:
        return super(CompactIOMachine).getinfo()


lsi_suffices = ['.json', '.gensimdict', '.gensimtfidf', '.gensimmodel.projection',
                '.gensimmodel', '.gensimmat', ]

class LSIModeler(GensimTopicModeler, CompactIOMachine):
    """
    This class facilitates the creation of LSI (latent semantic indexing) topic models,
    with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.

    This class extends :class:`GensimTopicModeler`.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            toweigh: bool = True,
            normalize: bool = True
    ):
        GensimTopicModeler.__init__(
            self,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            algorithm="lsi",
            toweigh=toweigh,
            normalize=normalize
        )
        CompactIOMachine.__init__(
            self, {'classifier': 'lsitopic'}, 'lsitopic', lsi_suffices
        )

    def getinfo(self) -> dict[str, Any]:
        return super(CompactIOMachine).getinfo()


rp_suffices = ['.json', '.gensimtfidf', '.gensimmodel', '.gensimmat', '.gensimdict']

class RPModeler(GensimTopicModeler, CompactIOMachine):
    """
    This class facilitates the creation of RP (random projection) topic models,
    with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.

    This class extends :class:`GensimTopicModeler`.
    """
    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            toweigh: bool = True,
            normalize: bool = True
    ):
        GensimTopicModeler.__init__(
            self,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            algorithm="rp",
            toweigh=toweigh,
            normalize=normalize
        )
        CompactIOMachine.__init__(
            self, {'classifier': 'rptopic'}, 'rptopic', rp_suffices
        )

    def getinfo(self) -> dict[str, Any]:
        return super(CompactIOMachine).getinfo()


def load_gensimtopicmodel(
        name: str,
        preprocessor: Optional[callable] = None,
        tokenizer: Optional[callable] = None,
        compact: bool = True
):
    """ Load the gensim topic modeler from files.

    :param name: name (if compact=True) or prefix (if compact=False) of the file path
    :param preprocessor: function that preprocesses the text. (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: a topic modeler
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: GensimTopicModeler
    """
    if compact:
        modelerdict = {'ldatopic': LDAModeler, 'lsitopic': LSIModeler, 'rptopic': RPModeler}
        classifier_name = str(get_model_classifier_name(name))

        topicmodeler = modelerdict[classifier_name](preprocessor=preprocessor, tokenizer=tokenizer)
        topicmodeler.load_compact_model(name)
        return topicmodeler
    else:
        topicmodeler = GensimTopicModeler(preprocessor=preprocessor, tokenizer=tokenizer)
        topicmodeler.loadmodel(name)
        return topicmodeler
