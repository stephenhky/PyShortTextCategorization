from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.regularizers import l2

# Paper: Yoon Kim, "Convolutional Neural Networks for Sentence Classification," arXiv:1408.5882 (2014).
# ref: https://gist.github.com/entron/b9bc61a74e7cadeb1fec
# ref: http://cs231n.github.io/convolutional-networks/
def CNNWordEmbed(nb_labels,
                 nb_filters=1200,
                 n_gram=2,
                 maxlen=15,
                 vecsize=300,
                 cnn_dropout=0.0,
                 final_activation='softmax',
                 dense_wl2reg=0.0):
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_length=maxlen - n_gram + 1))
    model.add(Flatten())
    model.add(Dense(nb_labels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg))
              )
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

# two layers of CNN, maxpooling, dense
def DoubleCNNWordEmbed(nb_labels,
                       nb_filters_1=1200,
                       nb_filters_2=600,
                       n_gram=2,
                       filter_length_2=10,
                       maxlen=15,
                       vecsize=300,
                       cnn_dropout_1=0.0,
                       cnn_dropout_2=0.0,
                       final_activation='softmax',
                       dense_wl2reg=0.0):
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters_1,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout_1 > 0.0:
        model.add(Dropout(cnn_dropout_1))
    model.add(Convolution1D(nb_filter=nb_filters_2,
                            filter_length=filter_length_2,
                            border_mode='valid',
                            activation='relu'))
    if cnn_dropout_2 > 0.0:
        model.add(Dropout(cnn_dropout_2))
    model.add(MaxPooling1D(pool_length=maxlen - n_gram -filter_length_2 + 1))
    model.add(Flatten())
    model.add(Dense(nb_labels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

# C-LSTM
# Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau,
# "A C-LSTM Neural Network for Text Classification", arXiv:1511.08630 (2015).
def CLSTMWordEmbed(nb_labels,
                   nb_filters=1200,
                   n_gram=2,
                   maxlen=15,
                   vecsize=300,
                   cnn_dropout=0.0,
                   nb_rnnoutdim=1200,
                   rnn_dropout=0.2,
                   final_activation='softmax',
                   dense_wl2reg=0.0
                   ):
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_length=maxlen - n_gram + 1))
    model.add(LSTM(nb_rnnoutdim))
    if rnn_dropout > 0.0:
        model.add(Dropout(rnn_dropout))
    model.add(Dense(nb_labels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg)
                    )
              )
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model