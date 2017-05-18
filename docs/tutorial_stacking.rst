Stacked Generalization
======================

"Stacking generates the members of the stacking ensemble using several learning algorithms and subsequently
uses another algorithm to learn how to combine their outputs." It combines the classification results
of several classifiers, and combines them.

Stacking is most commonly implemented using logistic regression.
Suppose there are *K* classifiers, and *l* output labels. Then the stacking generalization
is this logistic model:

:math:`P ( y=c | x) = \frac{1}{\exp\left( - \sum_{k=1}^{K} w_{kc} x_{kc} + b_c \right) + 1}`

Here we demonstrate the use of stacking of two classifiers.

Import the package, and employ the subject dataset as the training dataset.

>>> import shorttext
>>> subdict = shorttext.data.subjectkeywords()

Train a C-LSTM model.

>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')
>>> clstm_nnet = shorttext.classifiers.frameworks.CLSTMWordEmbed(len(subdict))
>>> clstm_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
>>> clstm_classifier.train(subdict, clstm_nnet)

A test of its classification:

>>> clstm_classifier.score('linear algebra')
{'mathematics': 1.0, 'physics': 3.3643366e-10, 'theology': 1.0713742e-13}
>>> clstm_classifier.score('topological soliton')
{'mathematics': 2.0036438e-11, 'physics': 1.0, 'theology': 4.4903334e-14}

And we train an SVM, with topic vectors as the input vectors. The topic model is LDA with 128 topics.

>>> # train the LDA topic model
>>> lda128 = shorttext.classifiers.LDAModeler()
>>> lda128.train(subdict, 128)
>>> # train the SVM classifier
>>> from sklearn.svm import SVC
>>> lda128_svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(lda128, SVC())
>>> lda128_svm_classifier.train(subdict)

A test of its classification:

>>>  lda128_svm_classifier.score('linear algebra')
{'mathematics': 1.0, 'physics': 0.0, 'theology': 0.0}
>>> lda128_svm_classifier.score('topological soliton')
{'mathematics': 0.0, 'physics': 1.0, 'theology': 0.0}

Then we can implement the stacked generalization using logistic regression by calling:

>>> stacker = shorttext.stack.LogisticStackedGeneralization(intermediate_classifiers={'clstm': clstm_classifier, 'lda128': lda128_svm_classifier})
>>> stacker.train(subdict)

Now the model is ready. As a result, we can do the stacked classification:

>>> stacker.score('linear algebra')
{'mathematics': 0.55439126, 'physics': 0.036988281, 'theology': 0.039665185}
>>> stacker.score('quantum mechanics')
{'mathematics': 0.059210967, 'physics': 0.55031472, 'theology': 0.04532773}
>>> stacker.score('topological dynamics')
{'mathematics': 0.17244603, 'physics': 0.19720334, 'theology': 0.035309207}
>>> stacker.score('christology')
 {'mathematics': 0.094574735, 'physics': 0.053406414, 'theology': 0.3797417}

The stacked generalization can be saved by calling:

>>> stacker.save_compact_model('/path/to/logitmodel.bin')

This only saves the stacked generalization model, but not the intermediate classifiers.
The reason for this is for allowing flexibility for users to supply their own algorithms,
as long as they have the `score` functions which output the same way as the classifiers
offered in this package. To load them, initialize it in the same way:

>>> stacker2 = shorttext.stack.LogisticStackedGeneralization(intermediate_classifiers={'clstm': clstm_classifier, 'lda128': lda128_svm_classifier})
>>> stacker2.load_compact_model('/path/to/logitmodel.bin')

Reference
---------

"Combining the Best of All Worlds," *Everything About Data Analytics*, WordPress (2016). [`WordPress
<https://datawarrior.wordpress.com/2016/06/19/combining-the-best-of-all-worlds/>`_]

David H. Wolpert, "Stacked Generalization," *Neural Netw* 5: 241-259 (1992).

M. Paz Sesmero, Agapito I. Ledezma, Araceli Sanchis, "Generating ensembles of heterogeneous classifiers using Stacked Generalization,"
*WIREs Data Mining and Knowledge Discovery* 5: 21-34 (2015).

Home: :doc:`index`