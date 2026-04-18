
from typing import Optional, Literal

from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2


# Codes were changed because of Keras.
# Keras 1 --> Keras 2: https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes


# Paper: Yoon Kim, "Convolutional Neural Networks for Sentence Classification," arXiv:1408.5882 (2014).
# ref: https://gist.github.com/entron/b9bc61a74e7cadeb1fec
# ref: http://cs231n.github.io/convolutional-networks/
def CNNWordEmbed(
        nb_labels: int,
        wvmodel: Optional[KeyedVectors] = None,
        nb_filters: int = 1200,
        n_gram: int = 2,
        maxlen: int = 15,
        vecsize: int = 300,
        cnn_dropout: float = 0.0,
        final_activation: Literal["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"] = "softmax",
        dense_wl2reg: float = 0.0,
        dense_bl2reg: float = 0.0,
        optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam"
) -> Model:
    """Create a CNN for word embeddings.

    Args:
        nb_labels: Number of class labels.
        wvmodel: Word embedding model. If provided, vecsize is extracted from it.
        nb_filters: Number of filters. Default: 1200.
        n_gram: N-gram (window size). Default: 2.
        maxlen: Maximum sentence length. Default: 15.
        vecsize: Embedding vector size. Default: 300.
        cnn_dropout: CNN dropout rate. Default: 0.0.
        final_activation: Final layer activation. Default: softmax.
        dense_wl2reg: L2 regularization for weights. Default: 0.0.
        dense_bl2reg: L2 regularization for bias. Default: 0.0.
        optimizer: Optimizer. Default: adam.

    Returns:
        Keras Sequential model.

    Reference:
        Yoon Kim, "Convolutional Neural Networks for Sentence Classification,"
        EMNLP 2014 (arXiv:1408.5882).
        https://arxiv.org/abs/1408.5882
    """
    if wvmodel is not None:
        vecsize = wvmodel.vector_size

    model = Sequential()
    model.add(Conv1D(filters=nb_filters,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_size=maxlen - n_gram + 1))
    model.add(Flatten())
    model.add(Dense(nb_labels, kernel_regularizer=l2(dense_wl2reg), bias_regularizer=l2(dense_bl2reg)))
    model.add(Activation(final_activation))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def DoubleCNNWordEmbed(
        nb_labels: int,
        wvmodel: Optional[KeyedVectors] = None,
        nb_filters_1: int = 1200,
        nb_filters_2: int = 600,
        n_gram: int = 2,
        filter_length_2: int = 10,
        maxlen: int = 15,
        vecsize: int = 300,
        cnn_dropout_1: float = 0.0,
        cnn_dropout_2: float = 0.0,
        final_activation: Literal["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"] = "softmax",
        dense_wl2reg: float = 0.0,
        dense_bl2reg: float = 0.0,
        optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = 'adam'
) -> Model:
    """Create a double-layer CNN for word embeddings.

    Args:
        nb_labels: Number of class labels.
        wvmodel: Word embedding model. If provided, vecsize is extracted from it.
        nb_filters_1: Filters for first layer. Default: 1200.
        nb_filters_2: Filters for second layer. Default: 600.
        n_gram: N-gram for first layer. Default: 2.
        filter_length_2: Window size for second layer. Default: 10.
        maxlen: Maximum sentence length. Default: 15.
        vecsize: Embedding vector size. Default: 300.
        cnn_dropout_1: Dropout for first layer. Default: 0.0.
        cnn_dropout_2: Dropout for second layer. Default: 0.0.
        final_activation: Final layer activation. Default: softmax.
        dense_wl2reg: L2 regularization for weights. Default: 0.0.
        dense_bl2reg: L2 regularization for bias. Default: 0.0.
        optimizer: Optimizer. Default: adam.

    Returns:
        Keras Sequential model.
    """
    if wvmodel is not None:
        vecsize = wvmodel.vector_size

    model = Sequential()
    model.add(Conv1D(filters=nb_filters_1,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(maxlen, vecsize)))
    if cnn_dropout_1 > 0.0:
        model.add(Dropout(cnn_dropout_1))
    model.add(Conv1D(filters=nb_filters_2,
                     kernel_size=filter_length_2,
                     padding='valid',
                     activation='relu'))
    if cnn_dropout_2 > 0.0:
        model.add(Dropout(cnn_dropout_2))
    model.add(MaxPooling1D(pool_size=maxlen - n_gram -filter_length_2 + 1))
    model.add(Flatten())
    model.add(Dense(nb_labels, kernel_regularizer=l2(dense_wl2reg), bias_regularizer=l2(dense_bl2reg)))
    model.add(Activation(final_activation))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def CLSTMWordEmbed(
        nb_labels: int,
        wvmodel: Optional[KeyedVectors] = None,
        nb_filters: int = 1200,
        n_gram: int = 2,
        maxlen: int = 15,
        vecsize: int = 300,
        cnn_dropout: float = 0.0,
        nb_rnnoutdim: int = 1200,
        rnn_dropout: int = 0.2,
        final_activation: Literal["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"] = "softmax",
        dense_wl2reg: float = 0.0,
        dense_bl2reg: float = 0.0,
        optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam"
) -> Model:
    """Create a C-LSTM model for word embeddings.

    Args:
        nb_labels: Number of class labels.
        wvmodel: Word embedding model. If provided, vecsize is extracted from it.
        nb_filters: Number of CNN filters. Default: 1200.
        n_gram: N-gram (window size). Default: 2.
        maxlen: Maximum sentence length. Default: 15.
        vecsize: Embedding vector size. Default: 300.
        cnn_dropout: CNN dropout rate. Default: 0.0.
        nb_rnnoutdim: LSTM output dimension. Default: 1200.
        rnn_dropout: LSTM dropout rate. Default: 0.2.
        final_activation: Final layer activation. Default: softmax.
        dense_wl2reg: L2 regularization for weights. Default: 0.0.
        dense_bl2reg: L2 regularization for bias. Default: 0.0.
        optimizer: Optimizer. Default: adam.

    Returns:
        Keras Sequential model.

    Reference:
        Chunting Zhou et al., "A C-LSTM Neural Network for Text Classification,"
        arXiv:1511.08630 (2015).
        https://arxiv.org/abs/1511.08630
    """
    if wvmodel is not None:
        vecsize = wvmodel.vector_size

    model = Sequential()
    model.add(Conv1D(filters=nb_filters,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_size=maxlen - n_gram + 1))
    model.add(LSTM(nb_rnnoutdim))
    if rnn_dropout > 0.0:
        model.add(Dropout(rnn_dropout))
    model.add(Dense(nb_labels, kernel_regularizer=l2(dense_wl2reg), bias_regularizer=l2(dense_bl2reg)))
    model.add(Activation(final_activation))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
