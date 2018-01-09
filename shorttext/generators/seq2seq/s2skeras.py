
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense

from shorttext.utils import compactmodel_io as cio

# Reference: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

kerasseq2seq_suffices = {'.h5', '.json', '_encoder.h5', '_encoder.json', '_decoder.h5', '_decoder.json'}

@cio.compactio({'classifier': 'kerasseq2seq'}, 'kerasseq2seq', kerasseq2seq_suffices)
class Seq2SeqWithKeras:
    def __init__(self, vecsize, latent_dim):
        self.vecsize = vecsize
        self.latent_dim = latent_dim

    def prepare_model(self):
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
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, encoder_input, decoder_input, decoder_output, batch_size=64, epochs=100):
        self.model.fit([encoder_input, decoder_input], decoder_output,
                       batch_size=batch_size,
                       epochs=epochs)

    def savemodel(self, prefix, final=False):
        # save whole model
        if final:
            self.model.save_weights(prefix+'.h5')
        else:
            self.model.save(prefix+'.h5')
        open(prefix+'.json', 'wb').write(self.model.to_json())

        # save encoder and decoder
        if final:
            self.encoder_model.save_weights(prefix+'_encoder.h5')
            self.decoder_model.save_weights(prefix + '_decoder.h5')
        else:
            self.encoder_model.save(prefix + '_encoder.h5')
            self.decoder_model.save(prefix+'_decoder.h5')
        open(prefix+'_encoder.json', 'wb').write(self.encoder_model.to_json())
        open(prefix+'_decoder.json', 'wb').write(self.decoder_model.to_json())

    def loadmodel(self, prefix):
        self.model = load_model(prefix+'.h5')
        self.encoder_model = load_model(prefix+'_encoder.h5')
        self.decoder_model = load_model(prefix+'_decoder.h5')
