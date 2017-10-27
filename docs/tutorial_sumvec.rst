Word-Embedding Cosine Similarity Classifier
===========================================

Sum of Embedded Vectors
-----------------------

Given a pre-trained word-embedding models like Word2Vec, a classifier
based on cosine similarities can be built, which is
:class:`shorttext.classifiers.SumEmbeddedVecClassifier`.
In training the data,
the embedded vectors in every word in that class are averaged. The
score for a given text to each class is the cosine similarity between the averaged
vector of the given text and the precalculated vector of that class.

A pre-trained Google Word2Vec model can be downloaded `here
<https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.

See: :doc:`tutorial_wordembed` .

Import the package:

>>> import shorttext

To load the Word2Vec model,

>>> from shorttext.utils import load_word2vec_model
>>> wvmodel = load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

Then we load a set of data:

>>> nihtraindata = shorttext.data.nihreports(sample_size=None)

Then initialize the classifier:

>>> classifier = shorttext.classifiers.SumEmbeddedVecClassifier(wvmodel)   # for Google model, the vector size is 300 (default: 100)
>>> classifier.train(nihtraindata)

This classifier takes relatively little time to train compared with others
in this package. Then we can perform classification:

>>> classifier.score('bioinformatics')

Or the result can be sorted and only the five top-scored results are displayed:

>>> sorted(classifier.score('stem cell research').items(), key=lambda item: item[1], reverse=True)[:5]
[('NIGMS', 0.44962596182682935),
 ('NIAID', 0.4494126990050461),
 ('NINDS', 0.43435236806719524),
 ('NIDCR', 0.43042338197002483),
 ('NHGRI', 0.42878346869968731)]
>>> sorted(classifier.score('bioinformatics').items(), key=lambda item: item[1], reverse=True)[:5]
[('NHGRI', 0.54200061864847038),
 ('NCATS', 0.49097267547279988),
 ('NIGMS', 0.47818129591411118),
 ('CIT', 0.46874987052158501),
 ('NLM', 0.46869259072562974)]
>>> sorted(classifier.score('cancer immunotherapy').items(), key=lambda item: item[1], reverse=True)[:5]
[('NCI', 0.53734097785976076),
 ('NIAID', 0.50616582142027433),
 ('NIDCR', 0.48596330887674788),
 ('NIDDK', 0.46875755765903215),
 ('NCCAM', 0.4642233792198418)]

The trained model can be saved:

>>> classifier.save_compact_model('/path/to/sumvec_nihdata_model.bin')

And with the same pre-trained Word2Vec model, this classifier can be loaded:

>>> classifier2 = shorttext.classifiers.load_sumword2vec_classifier(wvmodel, '/path/to/sumvec_nihdata_model.bin')

Appendix: Model I/O in Previous Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In previous versions of `shorttext`, :class:`shorttext.classifiers.SumEmbeddedVecClassifier` has a `savemodel` method,
which runs as follow:

>>> classifier.savemodel('/path/to/nihdata')

This produces the following file for this model:

::

    /path/to/nihdata_embedvecdict.pkl

It can be loaded by:

>>> classifier2 = shorttext.classifiers.load_sumword2vec_classifier(wvmodel, '/path/to/nihdata', compact=False)

Reference
---------

Michael Czerny, "Modern Methods for Sentiment Analysis," *District Data Labs (2015). [`DistrictDataLabs
<https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis>`_]

Home: :doc:`index`