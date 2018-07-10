Supervised Classification with Topics as Features
=================================================

Topic Vectors as Intermediate Feature Vectors
---------------------------------------------

To perform classification using bag-of-words (BOW) model as features,
`nltk` and `gensim` offered good framework. But the feature vectors
of short text represented by BOW can be very sparse. And the relationships
between words with similar meanings are ignored as well. One of the way to
tackle this is to use topic modeling, i.e. representing the words
in a topic vector. This package provides the following ways to model
the topics:

- LDA (Latent Dirichlet Allocation)
- LSI (Latent Semantic Indexing)
- RP (Random Projections)
- Autoencoder

With the topic representations, users can use any supervised learning
algorithm provided by `scikit-learn` to perform the classification.

Topic Models in `gensim`: LDA, LSI, and Random Projections
----------------------------------------------------------

This package supports three algorithms provided by `gensim`, namely, LDA, LSI, and
Random Projections, to do the topic modeling.

>>> import shorttext

First, load a set of training data (all NIH data in this example):

>>> trainclassdict = shorttext.data.nihreports(sample_size=None)

Initialize an instance of topic modeler, and use LDA as an example:

>>> topicmodeler = shorttext.generators.LDAModeler()

For other algorithms, user can use :class:`LSIModeler` for LSI or :class:`RPModeler`
for RP. Everything else is the same.
To train with 128 topics, enter:

>>> topicmodeler.train(trainclassdict, 128)

After the training is done, the user can retrieve the topic vector representation
with the trained model. For example,

>>> topicmodeler.retrieve_topicvec('stem cell research')

>>> topicmodeler.retrieve_topicvec('bioinformatics')

By default, the vectors are normalized. Another way to retrieve the topic vector
representation is as follow:

>>> topicmodeler['stem cell research']

>>> topicmodeler['bioinformatics']

In the training and the retrieval above, the same preprocessing process is applied.
Users can provide their own preprocessor while initiating the topic modeler.

Users can save the trained model by calling:

>>> topicmodeler.save_compact_model('/path/to/nihlda128.bin')

And the topic model can be retrieved by calling:

>>> topicmodeler2 = shorttext.generators.load_gensimtopicmodel('/path/to/nihlda128.bin')

While initialize the instance of the topic modeler, the user can also specify
whether to weigh the terms using tf-idf (term frequency - inverse document frequency).
The default is to weigh. To not weigh, initialize it as

>>> topicmodeler3 = shorttext.generators.GensimTopicModeler(toweigh=False)

Appendix: Model I/O in Previous Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For previous versions of `shorttext`, the trained models are saved by calling:

>>> topicmodeler.savemodel('/path/to/nihlda128')

However, we discourage users using this anymore, because the model I/O for various models
in gensim have been different. It produces errors.

All of them have to be present in order to be loaded. Note that the preprocessor is
not saved. To load the model, enter:

>>> topicmodeler2 = shorttext.classifiers.load_gensimtopicmodel('/path/to/nihlda128', compact=False)

AutoEncoder
-----------

Note: Previous version (<=0.2.1) of this autoencoder has a serious bug. Current version is
incompatible with the autoencoder of version <=0.2.1 .

Another way to find a new topic vector representation is to use the autoencoder, a neural network model
which compresses a vector representation into another one of a shorter (or longer, rarely though)
representation, by minimizing the difference between the input layer and the decoding layer.
For faster demonstration, use the subject keywords as the example dataset:

>>> subdict = shorttext.data.subjectkeywords()

To train such a model, we perform in a similar way with the LDA model (or LSI and random projections above):

>>> autoencoder = shorttext.generators.AutoencodingTopicModeler()
>>> autoencoder.train(subdict, 8)

After the training is done, the user can retrieve the encoded vector representation
with the trained autoencoder model. For example,

>>> autoencoder.retrieve_topicvec('linear algebra')

>>> autoencoder.retrieve_topicvec('path integral')

By default, the vectors are normalized. Another way to retrieve the topic vector
representation is as follow:

>>> autoencoder['linear algebra']

>>> autoencoder['path integral']

In the training and the retrieval above, the same preprocessing process is applied.
Users can provide their own preprocessor while initiating the topic modeler.

Users can save the trained models, by calling:

>>> autoencoder.save_compact_model('/path/to/sub_autoencoder8.bin')

And the model can be retrieved by calling:

>>> autoencoder2 = shorttext.generators.load_autoencoder_topicmodel('/path/to/sub_autoencoder8.bin')

Like other topic models, while initialize the instance of the topic modeler, the user can also specify
whether to weigh the terms using tf-idf (term frequency - inverse document frequency).
The default is to weigh. To not weigh, initialize it as:

>>> autoencoder3 = shorttext.generators.AutoencodingTopicModeler(toweigh=False)

Appendix: Unzipping Model I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For previous versions of `shorttext`, the trained models are saved by calling:

>>> autoencoder.savemodel('/path/to/sub_autoencoder8')

The following files are produced for the autoencoder:

::

    /path/to/sub_autoencoder.json
    /path/to/sub_autoencoder.gensimdict
    /path/to/sub_autoencoder_encoder.json
    /path/to/sub_autoencoder_encoder.h5
    /path/to/sub_autoencoder_classtopicvecs.pkl

If specifying `save_complete_autoencoder=True`, then four more files are found:

::

    /path/to/sub_autoencoder_decoder.json
    /path/to/sub_autoencoder_decoder.h5
    /path/to/sub_autoencoder_autoencoder.json
    /path/to/sub_autoencoder_autoencoder.h5

Users can load the same model later by entering:

>>> autoencoder2 = shorttext.classifiers.load_autoencoder_topic('/path/to/sub_autoencoder8', compact=False)


Abstract Latent Topic Modeling Class
------------------------------------

Both :class:`shorttext.generators.GensimTopicModeler` and
:class:`shorttext.generators.AutoencodingTopicModeler` extends
:class:`shorttext.generators.bow.LatentTopicModeling.LatentTopicModeler`,
an abstract class virtually. If user wants to develop its own topic model that extends
this, he has to define the methods `train`, `retrieve_topic_vec`, `loadmodel`, and
`savemodel`.

Appendix: Namespaces for Topic Modeler in Previous Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All generative topic modeling algorithms were placed under the package `shorttext.classifiers` for version <=0.3.4.
In current version (>= 0.3.5), however, all generative models will be moved to `shorttext.generators`,
while any classifiers making use of these topic models are still kept under `shorttext.classifiers`.
A list include:

::

    shorttext.classifiers.GensimTopicModeler  ->  shorttext.generators.GensimTopicModeler
    shorttext.classifiers.LDAModeler  ->  shorttext.generators.LDAModeler
    shorttext.classifiers.LSIModeler  ->  shorttext.generators.LSIModeler
    shorttext.classifiers.RPModeler  ->  shorttext.generators.RPModeler
    shorttext.classifiers.AutoencodingTopicModeler  ->  shorttext.generators.AutoencodingTopicModeler
    shorttext.classifiers.load_gensimtopicmodel  ->  shorttext.generators.load_gensimtopicmodel
    shorttext.classifiers.load_autoencoder_topic  ->  shorttext.generators.load_autoencoder_topicmodel


Before release 0.5.6, for backward compatibility, developers can still call the topic models as if there were no such changes,
although they are advised to make this change. However, *effective release 0.5.7, this backward compatibility is no longer
available.*

Classification Using Cosine Similarity
--------------------------------------

The topic modelers are trained to represent the short text in terms of a topic vector,
effectively the feature vector. However, to perform supervised classification, there
needs a classification algorithm. The first one is to calculate the cosine similarities
between topic vectors of the given short text with those of the texts in all class labels.

If there is already a trained topic modeler, whether it is
:class:`shorttext.generators.GensimTopicModeler` or
:class:`shorttext.generators.AutoencodingTopicModeler`,
a classifier based on cosine similarities can be initiated
immediately without training. Taking the LDA example above, such classifier can be initiated as follow:

>>> cos_classifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)

Or if the user already saved the topic modeler, one can initiate the same classifier by
loading the topic modeler:

>>> cos_classifier = shorttext.classifiers.load_gensimtopicvec_cosineClassifier('/path/to/nihlda128.bin')

To perform prediction, enter:

>>> cos_classifier.score('stem cell research')

which outputs a dictionary with labels and the corresponding scores.

The same thing for autoencoder, but the classifier based on autoencoder can be loaded by another function:

>>> cos_classifier = shorttext.classifiers.load_autoencoder_cosineClassifier('/path/to/sub_autoencoder8.bin')

Classification Using Scikit-Learn Classifiers
---------------------------------------------

The topic modeler can be used to generate features used for other machine learning
algorithms. We can take any supervised learning algorithms in `scikit-learn` here.
We use Gaussian naive Bayes as an example. For faster demonstration, use the subject
keywords as the example dataset.

>>> subtopicmodeler = shorttext.generators.LDAModeler()
>>> subtopicmodeler.train(subdict, 8)

We first import the class:

>>> from sklearn.naive_bayes import GaussianNB

And we train the classifier:

>>> classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(subtopicmodeler, GaussianNB())
>>> classifier.train(subdict)

Predictions can be performed like the following example:

>>> classifier.score('functional integral')

which outputs a dictionary with labels and the corresponding scores.

You can save the model by:

>>> classifier.save_compact_model('/path/to/sublda8nb.bin')

where the argument specifies the prefix of the path of the model files, including the topic
models, and the scikit-learn model files. The classifier can be loaded by calling:

>>> classifier2 = shorttext.classifiers.load_gensim_topicvec_sklearnclassifier('/path/to/sublda8nb.bin')

The topic modeler here can also be an autoencoder, by putting `subtopicmodeler` as the autoencoder
will still do the work. However, to load the saved classifier with an autoencoder model, do

>>> classifier2 = shorttext.classifiers.load_autoencoder_topic_sklearnclassifier('/path/to/filename.bin')

Compact model files saved by `TopicVectorSkLearnClassifier` in `shorttext` >= 1.0.0 cannot be read
by earlier version of `shorttext`; vice versa is not true though: old compact model files can be read in.

Notes about Text Preprocessing
------------------------------

The topic models are based on bag-of-words model, and text preprocessing is very important.
However, the text preprocessing step cannot be serialized. The users should keep track of the
text preprocessing step on their own. Unless it is necessary, use the standard preprocessing.

See more: :doc:`tutorial_textpreprocessing` .

Reference
---------

David M. Blei, "Probabilistic Topic Models," *Communications of the ACM* 55(4): 77-84 (2012).

Francois Chollet, "Building Autoencoders in Keras," *The Keras Blog*. [`Keras
<https://blog.keras.io/building-autoencoders-in-keras.html>`_]

Xuan Hieu Phan, Cam-Tu Nguyen, Dieu-Thu Le, Minh Le Nguyen, Susumu Horiguchi, Quang-Thuy Ha,
"A Hidden Topic-Based Framework toward Building Applications with Short Web Documents,"
*IEEE Trans. Knowl. Data Eng.* 23(7): 961-976 (2011).

Xuan Hieu Phan, Le-Minh Nguyen, Susumu Horiguchi, "Learning to Classify Short and Sparse Text & Web withHidden Topics from Large-scale Data Collections,"
WWW '08 Proceedings of the 17th international conference on World Wide Web. (2008) [`ACL
<http://dl.acm.org/citation.cfm?id=1367510>`_]

Home: :doc:`index`