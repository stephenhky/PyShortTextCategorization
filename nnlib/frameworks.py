from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2


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