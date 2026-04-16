
from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
import joblib
import sklearn

from ....utils import textpreprocessing as textpreprocess
from ....generators import load_autoencoder_topicmodel, load_gensimtopicmodel
from ....generators import LDAModeler, LSIModeler, RPModeler, AutoencodingTopicModeler
from ....generators import LatentTopicModeler
from ....utils import classification_exceptions as e
from ....utils import compactmodel_io as cio
from ...base import AbstractScorer


class TopicVectorSkLearnClassifier(AbstractScorer):
    """Classifier using topic vectors with scikit-learn.

    Wraps any scikit-learn supervised learning algorithm and uses
    topic vectors from LatentTopicModeler as features.

    Reference:
        Xuan Hieu Phan et al., "A Hidden Topic-Based Framework toward
        Building Applications with Short Web Documents,"
        IEEE Trans. Knowl. Data Eng. 23(7): 961-976 (2011).

        Xuan Hieu Phan et al., "Learning to Classify Short and Sparse
        Text & Web with Hidden Topics from Large-scale Data Collections,"
        WWW 2008.
        http://dl.acm.org/citation.cfm?id=1367510
    """

    def __init__(
            self,
            topicmodeler: LatentTopicModeler,
            sklearn_classifier: sklearn.base.BaseEstimator
    ):
        """Initialize the classifier.

        Args:
            topicmodeler: A topic modeler instance.
            sklearn_classifier: A scikit-learn classifier instance.
        """
        self.topicmodeler = topicmodeler
        self.classifier = sklearn_classifier
        self.trained = False

    def train(self, classdict: dict[str, list[str]], *args, **kwargs) -> None:
        """Train the classifier.

        Args:
            classdict: Training data with class labels as keys and texts as values.
            *args: Arguments passed to scikit-learn classifier fit().
            **kwargs: Arguments passed to scikit-learn classifier fit().

        Raises:
            ModelNotTrainedException: If topic modeler is not trained.
        """
        x = []
        y = []
        self.classlabels = sorted(classdict.keys())     # classlabels must be sorted like the topic modelers
        for classidx, classlabel in enumerate(self.classlabels):
            topicvecs = [
                self.topicmodeler.retrieve_topicvec(shorttext)
                for shorttext in classdict[classlabel]
            ]
            x += topicvecs
            y += [classidx]*len(topicvecs)
        self.classifier.fit(x, y, *args, **kwargs)
        self.trained = True

    def getvector(self, shorttext: str) -> npt.NDArray[np.float64]:
        """Get topic vector for short text.

        Args:
            shorttext: Input text.

        Returns:
            Topic vector representation.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        return self.topicmodeler.retrieve_topicvec(shorttext)

    def classify(self, shorttext: str) -> str:
        """Classify short text into a class label.

        Args:
            shorttext: Input text to classify.

        Returns:
            Predicted class label.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        topicvec = self.getvector(shorttext)
        return self.classlabels[self.classifier.predict([topicvec])[0]]

    def score(self, shorttext: str) -> dict[str, float]:
        """Compute classification scores for all classes.

        Args:
            shorttext: Input text.

        Returns:
            Dictionary mapping class labels to scores.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        topicvec = self.getvector(shorttext)
        scoredict = {
            classlabel: self.classifier.score([topicvec], [classidx])
            for classidx, classlabel in enumerate(self.classlabels)
        }
        return scoredict

    def savemodel(self, nameprefix: str) -> None:
        """Save model to files.

        Saves the topic model, scikit-learn classifier, and class labels.

        Args:
            nameprefix: Prefix for output files.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        self.topicmodeler.savemodel(nameprefix)
        joblib.dump(self.classifier, nameprefix+'.pkl')
        labelfile = open(nameprefix+'_classlabels.txt', 'w')
        labelfile.write('\n'.join(self.classlabels))
        labelfile.close()

    def loadmodel(self, nameprefix: str) -> None:
        """Load model from files.

        Args:
            nameprefix: Prefix for input files.
        """
        self.topicmodeler.loadmodel(nameprefix)
        self.classifier = joblib.load(nameprefix+'.pkl')
        labelfile = open(nameprefix+'_classlabels.txt', 'r')
        self.classlabels = [s.strip() for s in labelfile.readlines()]
        labelfile.close()

    def save_compact_model(self, name: str) -> None:
        """Save model as compact file.

        Args:
            name: Name of the compact model file.

        Raises:
            ModelNotTrainedException: If model not trained.
        """
        topicmodel_info = self.topicmodeler.get_info()
        cio.save_compact_model(
            name,
            self.savemodel,
            'topic_sklearn',
            topicmodel_info['suffices']+['.pkl', '_classlabels.txt'],
            {
                'classifier': 'topic_sklearn',
                'topicmodel': topicmodel_info['classifier']
            }
        )

    def load_compact_model(self, name: str) -> None:
        """Load model from compact file.

        Args:
            name: Name of the compact model file.
        """
        cio.load_compact_model(
            name,
            self.loadmodel,
            'topic_sklearn',
            {'classifier': 'topic_sklearn', 'topicmodel': None}
        )
        self.trained = True


def train_gensim_topicvec_sklearnclassifier(
        classdict: dict[str, list[str]],
        nb_topics: int,
        sklearn_classifier: sklearn.base.BaseEstimator,
        preprocessor: Optional[callable] = None,
        topicmodel_algorithm: Literal["lda", "lsi", "rp"] = 'lda',
        toweigh: bool = True,
        normalize: bool = True,
        gensim_paramdict: Optional[dict] = None,
        sklearn_paramdict: Optional[dict] = None
) -> TopicVectorSkLearnClassifier:
    """Train a classifier with gensim topic vectors and scikit-learn.

    Trains a topic model (LDA, LSI, or RP), then uses the topic vectors
    as features to train a scikit-learn classifier.

    Args:
        classdict: Training data.
        nb_topics: Number of topics.
        sklearn_classifier: Scikit-learn classifier instance (not trained).
        preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
        topicmodel_algorithm: Topic model algorithm. Default: lda.
        toweigh: Apply tf-idf weighting. Default: True.
        normalize: Normalize topic vectors. Default: True.
        gensim_paramdict: Arguments for gensim topic model.
        sklearn_paramdict: Arguments for scikit-learn classifier.

    Returns:
        Trained TopicVectorSkLearnClassifier.

    Reference:
        Xuan Hieu Phan et al., "A Hidden Topic-Based Framework toward
        Building Applications with Short Web Documents,"
        IEEE Trans. Knowl. Data Eng. 23(7): 961-976 (2011).

        Xuan Hieu Phan et al., "Learning to Classify Short and Sparse
        Text & Web with Hidden Topics from Large-scale Data Collections,"
        WWW 2008.
        http://dl.acm.org/citation.cfm?id=1367510
    """
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()
    if gensim_paramdict is None:
        gensim_paramdict = {}
    if sklearn_paramdict is None:
        sklearn_paramdict = {}

    modelerdict = {'lda': LDAModeler, 'lsi': LSIModeler, 'rp': RPModeler}
    topicmodeler = modelerdict[topicmodel_algorithm](
        preprocessor=preprocessor,
        toweigh=toweigh,
        normalize=normalize
    )
    topicmodeler.train(classdict, nb_topics, **gensim_paramdict)

    classifier = TopicVectorSkLearnClassifier(topicmodeler, sklearn_classifier)
    classifier.train(classdict, **sklearn_paramdict)

    return classifier


def load_gensim_topicvec_sklearnclassifier(
        name: str,
        preprocessor: Optional[callable] = None,
        compact: bool = True
) -> TopicVectorSkLearnClassifier:
    """Load a classifier with gensim topic vectors from files.

    Args:
        name: Model name (compact) or file prefix (non-compact).
        preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
        compact: Load compact model. Default: True.

    Returns:
        TopicVectorSkLearnClassifier instance.

    Reference:
        Xuan Hieu Phan et al., "A Hidden Topic-Based Framework toward
        Building Applications with Short Web Documents,"
        IEEE Trans. Knowl. Data Eng. 23(7): 961-976 (2011).

        Xuan Hieu Phan et al., "Learning to Classify Short and Sparse
        Text & Web with Hidden Topics from Large-scale Data Collections,"
        WWW 2008.
        http://dl.acm.org/citation.cfm?id=1367510
    """
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()

    if compact:
        modelerdict = {'ldatopic': LDAModeler, 'lsitopic': LSIModeler, 'rptopic': RPModeler}
        topicmodel_name = cio.get_model_config_field(name, 'topicmodel')
        classifier = TopicVectorSkLearnClassifier(modelerdict[topicmodel_name](preprocessor=preprocessor), None)
        classifier.load_compact_model(name)
        classifier.trained = True
        return classifier
    else:
        topicmodeler = load_gensimtopicmodel(name, preprocessor=preprocessor)
        sklearn_classifier = joblib.load(name + '.pkl')
        classifier = TopicVectorSkLearnClassifier(topicmodeler, sklearn_classifier)
        classifier.trained = True
        return classifier


def train_autoencoder_topic_sklearnclassifier(
        classdict: dict[str, list[str]],
        nb_topics: int,
        sklearn_classifier: sklearn.base.BaseEstimator,
        preprocessor: Optional[callable] = None,
        normalize: bool = True,
        keras_paramdict: Optional[dict] = None,
        sklearn_paramdict: Optional[dict] = None
) -> TopicVectorSkLearnClassifier:
    """Train a classifier with autoencoder topic vectors and scikit-learn.

    Trains an autoencoder topic model, then uses the encoded vectors
    as features to train a scikit-learn classifier.

    Args:
        classdict: Training data.
        nb_topics: Number of encoding dimensions.
        sklearn_classifier: Scikit-learn classifier instance (not trained).
        preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
        normalize: Normalize topic vectors. Default: True.
        keras_paramdict: Arguments for Keras autoencoder training.
        sklearn_paramdict: Arguments for scikit-learn classifier.

    Returns:
        Trained TopicVectorSkLearnClassifier.

    Reference:
        Xuan Hieu Phan et al., "A Hidden Topic-Based Framework toward
        Building Applications with Short Web Documents,"
        IEEE Trans. Knowl. Data Eng. 23(7): 961-976 (2011).

        Xuan Hieu Phan et al., "Learning to Classify Short and Sparse
        Text & Web with Hidden Topics from Large-scale Data Collections,"
        WWW 2008.
        http://dl.acm.org/citation.cfm?id=1367510
    """
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()
    if keras_paramdict is None:
        keras_paramdict = {}
    if sklearn_paramdict is None:
        sklearn_paramdict = {}

    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor, normalize=normalize)
    autoencoder.train(classdict, nb_topics, **keras_paramdict)

    classifier = TopicVectorSkLearnClassifier(autoencoder, sklearn_classifier)
    classifier.train(classdict, **sklearn_paramdict)

    return classifier


def load_autoencoder_topic_sklearnclassifier(
        name: str,
        preprocessor: Optional[callable] = None,
        compact: bool = True
) -> TopicVectorSkLearnClassifier:
    """Load a classifier with autoencoder topic vectors from files.

    Args:
        name: Model name (compact) or file prefix (non-compact).
        preprocessor: Text preprocessing function. Default: standard_text_preprocessor_1.
        compact: Load compact model. Default: True.

    Returns:
        TopicVectorSkLearnClassifier instance.

    Reference:
        Xuan Hieu Phan et al., "A Hidden Topic-Based Framework toward
        Building Applications with Short Web Documents,"
        IEEE Trans. Knowl. Data Eng. 23(7): 961-976 (2011).

        Xuan Hieu Phan et al., "Learning to Classify Short and Sparse
        Text & Web with Hidden Topics from Large-scale Data Collections,"
        WWW 2008.
        http://dl.acm.org/citation.cfm?id=1367510
    """
    if preprocessor is None:
        preprocessor = textpreprocess.standard_text_preprocessor_1()

    if compact:
        classifier = TopicVectorSkLearnClassifier(AutoencodingTopicModeler(preprocessor=preprocessor), None)
        classifier.load_compact_model(name)
        classifier.trained = True
        return classifier
    else:
        autoencoder = load_autoencoder_topicmodel(name, preprocessor=preprocessor)
        sklearn_classifier = joblib.load(name + '.pkl')
        classifier = TopicVectorSkLearnClassifier(autoencoder, sklearn_classifier)
        classifier.trained = True
        return classifier
