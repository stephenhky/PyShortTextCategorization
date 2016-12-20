Short Text Categorization Using Word-Embedding and Neural Networks
==================================================================

Wrapper for Neural Networks for Word-Embedding Vectors
------------------------------------------------------

In this package, there is a class that serves a wrapper for various neural network algorithms
for supervised short text categorization:
:class:`shorttext.classifiers.embed.nnlib.VarNNEmbeddedVecClassification.VarNNEmbeddedVecClassifier`.
Each class label has a few short sentences, where each token is converted
to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
The sentences are represented by a matrix, or rank-2 array.
The type of neural network has to be passed when training, and it has to be of
type :class:`keras.models.Sequential`. The number of outputs of the models has to match
the number of class labels in the training data.
To perform prediction, the input short sentences is converted to a unit vector
in the same way. The score is calculated according to the trained neural network model.

Some of the neural networks can be found within the module :module:`shorttext.classifiers.embed.nnlib.frameworks`
and they are good for short text or document classification. Of course, users can supply their
own neural networks, written in `keras`.

A pre-trained Google Word2Vec model can be downloaded `here
<https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.

See: :doc:`tutorial_wordembed` .

Import the package:

>>> import shorttext

To load the Word2Vec model,

>>> from shorttext.utils.wordembed import load_word2vec_model
>>> wvmodel = load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

Then load the training data
>>> trainclassdict = shorttext.data.subjectkeywords()

Then we choose a neural network. We choose ConvNet:

>>> import shorttext.classifiers.embed.nnlib.frameworks as fr
>>> kmodel = fr.CNNWordEmbed(len(trainclassdict.keys()))

Initialize the classifier:

>>> import shorttext.classifiers.embed.nnlib.VarNNEmbedVecClassification as vnn
>>> classifier = vnn.VarNNEmbeddedVecClassifier(wvmodel)

Then train the classifier:

>>> classifier.train(trainclassdict, kmodel)
Epoch 1/10
45/45 [==============================] - 0s - loss: 1.0578
Epoch 2/10
45/45 [==============================] - 0s - loss: 0.5536
Epoch 3/10
45/45 [==============================] - 0s - loss: 0.3437
Epoch 4/10
45/45 [==============================] - 0s - loss: 0.2282
Epoch 5/10
45/45 [==============================] - 0s - loss: 0.1658
Epoch 6/10
45/45 [==============================] - 0s - loss: 0.1273
Epoch 7/10
45/45 [==============================] - 0s - loss: 0.1052
Epoch 8/10
45/45 [==============================] - 0s - loss: 0.0961
Epoch 9/10
45/45 [==============================] - 0s - loss: 0.0839
Epoch 10/10
45/45 [==============================] - 0s - loss: 0.0743

Then the model is ready for classification, like:

>>> classifier.score('artificial intelligence')
{'mathematics': 0.57749695, 'physics': 0.33749574, 'theology': 0.085007325}

Provided Neural Networks
------------------------

There are three neural networks available in this package for the use in
:class:`shorttext.classifiers.embed.nnlib.VarNNEmbeddedVecClassification.VarNNEmbeddedVecClassifier`,
and they are available in the module :module:`shorttext.classifiers.embed.nnlib.frameworks`.

ConvNet (Convolutional Neural Network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This neural network for supervised learning is using convolutional neural network (ConvNet),
as demonstrated in Kim's paper.

.. image:: images/nnlib_cnn.png

The function in the frameworks returns a :class:`keras.models.Sequential`.

.. autofunction:: shorttext.classifiers.embed.nnlib.frameworks.CNNWordEmbed

The parameter `maxlen` defines the maximum length of the sentences. If the sentence has less than `maxlen`
words, then the empty words will be filled with zero vectors.

>>> kmodel = fr.CNNWordEmbed(len(trainclassdict.keys()))

Double ConvNet
^^^^^^^^^^^^^^

This neural network is nothing more than two ConvNet layers.

.. autofunction:: shorttext.classifiers.embed.nnlib.frameworks.DoubleCNNWordEmbed

The parameter `maxlen` defines the maximum length of the sentences. If the sentence has less than `maxlen`
words, then the empty words will be filled with zero vectors.

>>> kmodel = fr.DoubleCNNWordEmbed(len(trainclassdict.keys()))

C-LSTM (Convolutional Long Short-Term Memory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This neural network for supervised learning is using C-LSTM, according to the paper
written by Zhou *et. al.* It is a neural network with ConvNet as the first layer,
and then followed by LSTM (long short-term memory), a type of recurrent neural network (RNN).

.. image:: images/nnlib_clstm.png

The function in the frameworks returns a :class:`keras.models.Sequential`.

.. autofunction:: shorttext.classifiers.embed.nnlib.frameworks.CLSTMWordEmbed

The parameter `maxlen` defines the maximum length of the sentences. If the sentence has less than `maxlen`
words, then the empty words will be filled with zero vectors.

>>> kmodel = fr.CLSTMWordEmbed(len(trainclassdict.keys()))

User-Defined Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can define their own neural network for use in the classifier wrapped by
:class:`shorttext.classifiers.embed.nnlib.VarNNEmbeddedVecClassification.VarNNEmbeddedVecClassifier`
as long as the following criteria are met:

- the input matrix is :class:`numpy.ndarray`, and of shape `(maxlen, vecsize)`, where
`maxlen` is the maximum length of the sentence, and `vecsize` is the number of dimensions
of the embedded vectors. The output is a one-dimensional array, of size equal to
the number of classes provided by the training data. The order of the class labels is assumed
to be the same as the order of the given training data (stored as a Python dictionary).

Reference
---------

Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau, "A C-LSTM Neural Network for Text Classification," (arXiv:1511.08630). [`arXiv
<https://arxiv.org/abs/1511.08630>`_]

"CS231n Convolutional Neural Networks for Visual Recognition," Stanford Online Course. [`link
<http://cs231n.github.io/convolutional-networks/>`_]

Yoon Kim, "Convolutional Neural Networks for Sentence Classification," *EMNLP* 2014, 1746-1751 (arXiv:1408.5882). [`arXiv
<https://arxiv.org/abs/1408.5882>`_]

Zackary C. Lipton, John Berkowitz, "A Critical Review of Recurrent Neural Networks for Sequence Learning," arXiv:1506.00019 (2015). [`arXiv
<https://arxiv.org/abs/1506.00019>`_]