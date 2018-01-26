
import json

import numpy as np
import gensim

from .s2skeras import Seq2SeqWithKeras, loadSeq2SeqWithKeras, kerasseq2seq_suffices
from ..charbase.char2vec import SentenceToCharVecEncoder
from shorttext.utils import compactmodel_io as cio


charbases2s_suffices = kerasseq2seq_suffices + ['_dictionary.dict', '_charbases2s.json']


@cio.compactio({'classifier': 'charbases2s'}, 'charbases2s', charbases2s_suffices)
class CharBasedSeq2SeqGenerator:
    def __init__(self, sent2charvec_encoder, latent_dim, maxlen):
        self.compiled = False
        if sent2charvec_encoder != None:
            self.sent2charvec_encoder = sent2charvec_encoder
            self.dictionary = self.sent2charvec_encoder.dictionary
            self.nbelem = len(self.dictionary)
            self.latent_dim = latent_dim
            self.maxlen = maxlen
            self.s2sgenerator = Seq2SeqWithKeras(self.nbelem, self.latent_dim)

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        if not self.compiled:
            self.s2sgenerator.prepare_model()
            self.s2sgenerator.compile(optimizer=optimizer, loss=loss)
            self.compiled = True

    def prepare_trainingdata(self, txtseq):
        encoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[:-1], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[1:], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_output = self.sent2charvec_encoder.encode_sentences(txtseq[1:], endsig=True, maxlen=self.maxlen, sparse=False)
        return encoder_input, decoder_input, decoder_output

    def train(self, txtseq, batch_size=64, epochs=100, optimizer='rmsprop', loss='categorical_crossentropy'):
        encoder_input, decoder_input, decoder_output = self.prepare_trainingdata(txtseq)
        self.compile(optimizer=optimizer, loss=loss)
        self.s2sgenerator.fit(encoder_input, decoder_input, decoder_output, batch_size=batch_size, epochs=epochs)

    def decode(self, txtseq):
        # Encode the input as state vectors.
        inputvec = np.array([self.sent2charvec_encoder.encode_sentence(txtseq, maxlen=self.maxlen, endsig=True).toarray()])
        states_value = self.s2sgenerator.encoder_model.predict(inputvec)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.nbelem))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.dictionary.token2id['\n']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_txtseq = ''
        while not stop_condition:
            output_tokens, h, c = self.s2sgenerator.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.dictionary[sampled_token_index]
            decoded_txtseq += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_txtseq) > self.maxlen):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.nbelem))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_txtseq

    def savemodel(self, prefix, final=False):
        self.s2sgenerator.savemodel(prefix, final=final)
        self.dictionary.save(prefix+'_dictionary.dict')
        json.dump({'maxlen': self.maxlen, 'latent_dim': self.latent_dim}, open(prefix+'_charbases2s.json', 'wb'))

    def loadmodel(self, prefix):
        self.dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
        self.s2sgenerator = loadSeq2SeqWithKeras(prefix, compact=False)
        self.sent2charvec_encoder = SentenceToCharVecEncoder(self.dictionary)
        self.nbelem = len(self.dictionary)
        hyperparameters = json.load(open(prefix+'_charbases2s.json', 'rb'))
        self.latent_dim, self.maxlen = hyperparameters['latent_dim'], hyperparameters['maxlen']
        self.compiled = True

def loadCharBasedSeq2SeqGenerator(path, compact=True):
    seq2seqer = CharBasedSeq2SeqGenerator(None, 0, 0)
    if compact:
        seq2seqer.load_compact_model(path)
    else:
        seq2seqer.loadmodel(path)
    return seq2seqer