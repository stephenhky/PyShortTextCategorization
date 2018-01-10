

from .s2skeras import Seq2SeqWithKeras
from ..charbase.char2vec import SentenceToCharVecEncoder


class CharBasedSeq2SeqGenerator:
    def __init__(self, dictionary, latent_dim):
        Seq2SeqWithKeras.__init__(self, len(dictionary), latent_dim)
        self.dictionary = dictionary
        self.nbelem = len(self.dictionary)
        self.latent_dim = latent_dim
        self.sent2charvec_encoder = SentenceToCharVecEncoder(self.dictionary)
        self.s2sgenerator = Seq2SeqWithKeras(self.nbelem, self.latent_dim)

