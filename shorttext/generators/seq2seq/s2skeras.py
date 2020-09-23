
import json

from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense

from shorttext.utils import compactmodel_io as cio
import shorttext.utils.classification_exceptions as e

# Reference: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

kerasseq2seq_suffices = ['.h5', '.json', '_s2s_hyperparam.json', '_encoder.h5', '_encoder.json', '_decoder.h5', '_decoder.json']


class Seq2SeqWithKeras(cio.CompactIOMachine):
    """ Class implementing sequence-to-sequence (seq2seq) learning with keras.

    Reference:

    Ilya Sutskever, James Martens, Geoffrey Hinton, "Generating Text with Recurrent Neural Networks," *ICML* (2011). [`UToronto
    <http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf>`_]

    Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks," arXiv:1409.3215 (2014). [`arXiv
    <https://arxiv.org/abs/1409.3215>`_]

    Francois Chollet, "A ten-minute introduction to sequence-to-sequence learning in Keras," *The Keras Blog*. [`Keras
    <https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html>`_]

    Aurelien Geron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (Sebastopol, CA: O'Reilly Media, 2017). [`O\'Reilly
    <http://shop.oreilly.com/product/0636920052289.do>`_]
    """
    def __init__(self, vecsize, latent_dim):
        """ Instantiate the class.

        :param vecsize: vector size of the sequence
        :param latent_dim: latent dimension in the RNN cell
        :type vecsize: int
        :type latent_dim: int
        """
        cio.CompactIOMachine.__init__(self, {'classifier': 'kerasseq2seq'}, 'kerasseq2seq', kerasseq2seq_suffices)
        self.vecsize = vecsize
        self.latent_dim = latent_dim
        self.compiled = False
        self.trained = False

    def prepare_model(self):
        """ Prepare the keras model.

        :return: None
        """
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.vecsize))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.vecsize))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.vecsize, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Compile the keras model after preparation running :func:`~prepare_model`.

        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :type optimizer: str
        :type loss: str
        :return: None
        """
        self.model.compile(optimizer=optimizer, loss=loss)
        self.compiled = True

    def fit(self, encoder_input, decoder_input, decoder_output, batch_size=64, epochs=100):
        """ Fit the sequence to learn the sequence-to-sequence (seq2seq) model.

        :param encoder_input: encoder input, a rank-3 tensor
        :param decoder_input: decoder input, a rank-3 tensor
        :param decoder_output: decoder output, a rank-3 tensor
        :param batch_size: batch size (Default: 64)
        :param epochs: number of epochs (Default: 100)
        :return: None
        :type encoder_input: numpy.array
        :type decoder_input: numpy.array
        :type decoder_output: numpy.array
        :type batch_size: int
        :type epochs: int
        """
        self.model.fit([encoder_input, decoder_input], decoder_output,
                       batch_size=batch_size,
                       epochs=epochs)
        self.trained = True

    def savemodel(self, prefix, final=False):
        """ Save the trained models into multiple files.

        To save it compactly, call :func:`~save_compact_model`.

        If `final` is set to `True`, the model cannot be further trained.

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param prefix: prefix of the file path
        :param final: whether the model is final (that should not be trained further) (Default: False)
        :return: None
        :type prefix: str
        :type final: bool
        :raise: ModelNotTrainedException
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        # save hyperparameters
        json.dump({'vecsize': self.vecsize, 'latent_dim': self.latent_dim}, open(prefix+'_s2s_hyperparam.json', 'w'))

        # save whole model
        if final:
            self.model.save_weights(prefix+'.h5')
        else:
            self.model.save(prefix+'.h5')
        open(prefix+'.json', 'w').write(self.model.to_json())

        # save encoder and decoder
        if final:
            self.encoder_model.save_weights(prefix+'_encoder.h5')
            self.decoder_model.save_weights(prefix + '_decoder.h5')
        else:
            self.encoder_model.save(prefix + '_encoder.h5')
            self.decoder_model.save(prefix+'_decoder.h5')
        open(prefix+'_encoder.json', 'w').write(self.encoder_model.to_json())
        open(prefix+'_decoder.json', 'w').write(self.decoder_model.to_json())

    def loadmodel(self, prefix):
        """ Load a trained model from various files.

        To load a compact model, call :func:`~load_compact_model`.

        :param prefix: prefix of the file path
        :return: None
        :type prefix: str
        """
        hyperparameters = json.load(open(prefix+'_s2s_hyperparam.json', 'r'))
        self.vecsize, self.latent_dim = hyperparameters['vecsize'], hyperparameters['latent_dim']
        self.model = load_model(prefix+'.h5')
        self.encoder_model = load_model(prefix+'_encoder.h5')
        self.decoder_model = load_model(prefix+'_decoder.h5')
        self.trained = True


def loadSeq2SeqWithKeras(path, compact=True):
    """ Load a trained `Seq2SeqWithKeras` class from file.

    :param path: path of the model file
    :param compact: whether it is a compact model (Default: True)
    :return: a `Seq2SeqWithKeras` class for sequence to sequence inference
    :type path: str
    :type compact: bool
    :rtype: Seq2SeqWithKeras
    """
    generator = Seq2SeqWithKeras(0, 0)
    if compact:
        generator.load_compact_model(path)
    else:
        generator.loadmodel(path)
    generator.compiled = True
    return generator