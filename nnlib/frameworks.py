from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2

# Paper: Yoon Kim, "Convolutional Neural Networks for Sentence Classification," arXiv:1408.5882 (2014).
#
# ref: https://gist.github.com/entron/b9bc61a74e7cadeb1fec
# ref: http://cs231n.github.io/convolutional-networks/
def CNNWordEmbed(numlabels,
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
    model.add(Dense(numlabels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg))
              )
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def DoubleCNNWordEmbed(numlabels,
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
    model.add(Dense(numlabels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
