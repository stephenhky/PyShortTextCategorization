Supervised Classification with Topics as Features
=================================================

To perform classification using bag-of-words (BOW) model as features,
`nltk` and `gensim` offered good framework. But the feature vectors
of short text represented by BOW can be very sparse. And the relationships
between words with similar meanings are ignored as well. One of the way to
tackle this is to use topic modeling, i.e. representing the words
in a topic vector. This package provides the following ways to model
the topics:

- LDA (Latent Dirichlet Allocation)
- LSI (Latent Semantic Indexing)
- Random Projections
- Autoencoder

With the topic representations, users can use any supervised learning
algorithm provided by `scikit-learn` to perform the classification.

LDA, LSI, and Random Projections
--------------------------------

This package supports three algorithms provided by `gensim`, namely, LDA, LSI, and
Random Projections, to do the topic modeling.

>>> import shorttext
>>> import shorttext.classifiers.bow.topic.LatentTopicModeling as ltm

First, load a set of training data (all NIH data in this example):

>>> trainclassdict = shorttext.data.nihreports(sample_size=None)





Reference
---------

Xuan Hieu Phan, Cam-Tu Nguyen, Dieu-Thu Le, Minh Le Nguyen, Susumu Horiguchi, Quang-Thuy Ha,
"A Hidden Topic-Based Framework toward Building Applications with Short Web Documents,"
*IEEE Trans. Knowl. Data Eng.* 23(7): 961-976 (2011).

Xuan Hieu Phan, Le-Minh Nguyen, Susumu Horiguchi, "Learning to Classify Short and Sparse Text & Web withHidden Topics from Large-scale Data Collections,"
WWW '08 Proceedings of the 17th international conference on World Wide Web. (2008) [`ACL
<http://dl.acm.org/citation.cfm?id=1367510>`_]