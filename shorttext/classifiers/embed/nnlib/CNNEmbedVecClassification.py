import VarNNEmbedVecClassification as vnn
import frameworks as fr


# wrapper for VarNNEmbedVecClassification for using CNN only
# backward compatible with the previous version of this class
class CNNEmbeddedVecClassifier:
    """
    This class is a wrapper that runs :class:`~vnn.VarNNEmbedVecClassification.VarNNEmbeddedVecClassifier`,
    but carries backward compatibility.
    """
    def __init__(self,
                 wvmodel,
                 classdict=None,
                 n_gram=2,
                 vecsize=300,
                 nb_filters=1200,
                 maxlen=15,
                 final_activation='softmax',
                 cnn_dropout=0.0,
                 dense_wl2reg=0.0,
                 optimizer='adam'):
        self.wvmodel = wvmodel
        self.vecsize = vecsize
        self.maxlen = maxlen
        self.trained = False
        self.wrapped_classifier = vnn.VarNNEmbeddedVecClassifier(self.wvmodel,
                                                                 vecsize=self.vecsize,
                                                                 maxlen=self.maxlen)

        self.classdict = classdict

        self.n_gram = n_gram
        self.nb_filters = nb_filters
        self.final_activation = final_activation
        self.cnn_dropout = cnn_dropout
        self.dense_wl2reg = dense_wl2reg
        self.optimizer = optimizer

    def train(self):
        self.kerasmodel = fr.CNNWordEmbed(len(self.classdict.keys()),
                                          nb_filters=self.nb_filters,
                                          n_gram=self.n_gram,
                                          maxlen=self.maxlen,
                                          vecsize=self.vecsize,
                                          cnn_dropout=self.cnn_dropout,
                                          final_activation=self.final_activation,
                                          dense_wl2reg=self.dense_wl2reg,
                                          optimizer=self.optimizer)

        self.wrapped_classifier.train(self.classdict, self.kerasmodel)

    def savemodel(self, nameprefix):
        self.wrapped_classifier.savemodel(nameprefix)

    def loadmodel(self, nameprefix):
        self.wrapped_classifier.loadmodel(nameprefix)

    def score(self, shorttext):
        return self.wrapped_classifier.score(shorttext)