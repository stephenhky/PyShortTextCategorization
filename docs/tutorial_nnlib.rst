Short Text Categorization Using Word-Embedding and Neural Networks
==================================================================

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
>>> kmodel = fr.CNNWordEmbed(len(classdict.keys()))

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


Reference
---------

Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau, "A C-LSTM Neural Network for Text Classification," (arXiv:1511.08630). [`arXiv
<https://arxiv.org/abs/1511.08630>`_]

Yoon Kim, "Convolutional Neural Networks for Sentence Classification," *EMNLP* 2014, 1746-1751 (arXiv:1408.5882). [`arXiv
<https://arxiv.org/abs/1408.5882>`_]
