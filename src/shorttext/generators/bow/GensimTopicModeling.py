
from typing import Optional, Literal, Any

import gensim
import numpy as np
import numpy.typing as npt
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, LsiModel, RpModel
from gensim.similarities import MatrixSimilarity
import orjson

from ...utils.classification_exceptions import ModelNotTrainedException
from ...utils.compactmodel_io import CompactIOMachine, get_model_classifier_name
from ...utils import gensim_corpora as gc
from .LatentTopicModeling import LatentTopicModeler


gensim_topic_model_dict = {'lda': LdaModel, 'lsi': LsiModel, 'rp': RpModel}


class GensimTopicModeler(LatentTopicModeler):
    """Topic modeler using gensim implementations.

    Supports LDA (Latent Dirichlet Allocation), LSI (Latent Semantic Indexing),
    and Random Projections (RP) for topic modeling.

    Note:
        For compact model I/O, use LDAModeler or LSIModeler instead.
    """

    def __init__(
            self,
            preprocessor: Optional[callable] = None,
            tokenizer: Optional[callable] = None,
            algorithm: Literal["lda", "lsi", "rp"] = "lda",
            toweigh: bool = True,
            normalize: bool = True
    ):
        """Initialize the topic modeler.

        Args:
            preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
            algorithm: Topic modeling algorithm. Options: 'lda', 'lsi', 'rp'. Default: 'lda'.
            toweigh: Whether to apply tf-idf weighting. Default: True.
            normalize: Whether to normalize topic vectors. Default: True.
        """
        LatentTopicModeler.__init__(
            self, preprocessor=preprocessor, tokenizer=tokenizer, normalize=normalize
        )
        self.algorithm = algorithm
        self.toweigh = toweigh

    def generate_corpus(self, classdict: dict[str, list[str]]) -> None:
        """Generate gensim dictionary and corpus.

        Args:
            classdict: Training data.
        """
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(
            classdict,
            preprocess_and_tokenize=lambda sent: self.tokenize_func(self.preprocess_func(sent))
        )

    def train(self, classdict: dict[str, list[str]], nb_topics: int, *args, **kwargs) -> None:
        """Train the topic modeler.

        Args:
            classdict: Training data with class labels as keys and texts as values.
            nb_topics: Number of latent topics.
            *args: Arguments for the gensim topic model.
            **kwargs: Keyword arguments for the gensim topic model.
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
        """Update model with additional data.

        Warning: Does not support adding new class labels or new vocabulary.
        For comprehensive updates, retrain the model.

        Args:
            additional_classdict: Additional training data.
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
        """Get bag-of-words representation.

        Args:
            shorttext: Input text.

        Returns:
            List of (word_id, count) tuples.
        """
        return self.dictionary.doc2bow(self.tokenize_func(self.preprocess_func(shorttext)))

    def retrieve_bow_vector(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get bag-of-words vector.

        Args:
            shorttext: Input text.

        Returns:
            BOW vector.
        """
        bow = self.retrieve_bow(shorttext)
        if len(bow) > 0:
            vec = np.zeros(len(self.dictionary))
            for id, val in bow:
                vec[id] = val
        else:
            vec = np.ones(len(self.dictionary))
        if self.normalize:
            vec /= np.linalg.norm(vec)
        return vec

    def retrieve_corpus_topicdist(self, shorttext: str) -> list[tuple[int, int | float]]:
        """Get topic distribution (corpus form).

        Args:
            shorttext: Input text.

        Returns:
            List of (topic_id, weight) tuples.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        bow = self.retrieve_bow(shorttext)
        return self.topicmodel[self.tfidf[bow] if self.toweigh else bow]

    def retrieve_topicvec(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get topic vector for short text.

        Args:
            shorttext: Input text.

        Returns:
            Topic vector.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        topicdist = self.retrieve_corpus_topicdist(shorttext)
        if len(topicdist) > 0:
            topicvec = np.zeros(self.nb_topics)
            for topicid, frac in topicdist:
                topicvec[topicid] = frac
        else:
            topicvec = np.ones(self.nb_topics)
        if self.normalize:
            topicvec /= np.linalg.norm(topicvec)
        return topicvec

    def get_batch_cos_similarities(self, shorttext: str) -> dict[str, float]:
        """Get cosine similarities to all classes.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to similarity scores.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()
        simdict = {}
        similarities = self.matsim[self.retrieve_corpus_topicdist(shorttext)]
        for label, similarity in zip(self.classlabels, similarities):
            simdict[label] = float(similarity)
        return simdict

    def loadmodel(self, nameprefix: str) -> None:
        """Load topic model from files.

        Args:
            nameprefix: Prefix for input files.
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
        """Save topic model to files.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise ModelNotTrainedException()

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

    def get_info(self) -> dict[str, Any]:
        return {}


lda_suffices =  [
    '.json', '.gensimdict', '.gensimmodel.state', '.gensimtfidf', '.gensimmodel',
    '.gensimmat', '.gensimmodel.expElogbeta.npy', '.gensimmodel.id2word'
]


class LDAModeler(GensimTopicModeler, CompactIOMachine):
    """LDA topic modeler with compact I/O support."""

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

    def get_info(self) -> dict[str, Any]:
        return CompactIOMachine.get_info(self)


lsi_suffices = ['.json', '.gensimdict', '.gensimtfidf', '.gensimmodel.projection',
                '.gensimmodel', '.gensimmat']

class LSIModeler(GensimTopicModeler, CompactIOMachine):
    """LSI topic modeler with compact I/O support."""

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

    def get_info(self) -> dict[str, Any]:
        return CompactIOMachine.get_info(self)


rp_suffices = ['.json', '.gensimtfidf', '.gensimmodel', '.gensimmat', '.gensimdict']

class RPModeler(GensimTopicModeler, CompactIOMachine):
    """Random Projection topic modeler with compact I/O support."""

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

    def get_info(self) -> dict[str, Any]:
        return CompactIOMachine.get_info(self)


def load_gensimtopicmodel(
        name: str,
        preprocessor: Optional[callable] = None,
        tokenizer: Optional[callable] = None,
        compact: bool = True
) -> GensimTopicModeler:
    """Load a gensim topic model from files.

    Args:
        name: Model name (compact) or file prefix (non-compact).
        preprocessor: Text preprocessing function.
        compact: Whether to load compact model. Default: True.

    Returns:
        A topic modeler instance.
    """
    if compact:
        modeler_dict = {'ldatopic': LDAModeler, 'lsitopic': LSIModeler, 'rptopic': RPModeler}
        classifier_name = str(get_model_classifier_name(name))
        if classifier_name not in modeler_dict.keys():
            raise ValueError(f"Unknown classifier name: {classifier_name}")

        topic_modeler = modeler_dict[classifier_name](preprocessor=preprocessor, tokenizer=tokenizer)
        topic_modeler.load_compact_model(name)
    else:
        modeler_dict = {'lda': LDAModeler, 'lsi': LSIModeler, 'rp': RPModeler}

        config_info = orjson.loads(open(name+".json", "rb").read())
        algorithm_name = config_info.get("algorithm")
        if algorithm_name is None:
            raise ValueError("No classifier name!")
        if algorithm_name not in modeler_dict.keys():
            raise ValueError(f"Unknown classifier name: {algorithm_name}")

        topic_modeler = modeler_dict[algorithm_name](preprocessor=preprocessor, tokenizer=tokenizer)
        topic_modeler.loadmodel(name)

    return topic_modeler
