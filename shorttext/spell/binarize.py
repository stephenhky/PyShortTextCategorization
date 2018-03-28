
import re
import string

import numpy as np
from shorttext.generators.charbase.char2concatvec import SpellingToConcatCharVecEncoder

default_alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
# NB. # is <eos>, _ is <unk>, @ is number
default_specialsignals = {'eos': '#', 'unk': '_', 'number': '@'}
default_signaldenotions = {'<eos>': 'eos', '<unk>': 'unk'}

def hasnum(word):
    return len(re.findall('\\d', word)) > 0

class SCRNNBinarizer:
    def __init__(self, alpha, signalchar_dict):
        self.signalchar_dict = signalchar_dict
        self.concatchar_encoder = SpellingToConcatCharVecEncoder(alpha)

    def noise_char(self, word, opt):
        bin_all = np.zeros((len(self.signalchar_dict), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_all[default_specialsignals[default_signaldenotions[word]]] += 1
        elif hasnum(word):
            bin_all[default_specialsignals['number']] += 1
        elif opt=='DELETE':
            idx = np.random.randint(0, len(word))
            w = word[:idx] + word[(idx+1):]
            bin_all = self.concatchar_encoder.encode_spelling(w)
        elif opt=='INSERT':
            ins_idx = np.random.randint(0, len(word)+1)
            ins_char = np.random.choice([c for c in string.ascii_lowercase])
            w = word[:ins_idx] + ins_char + word[ins_idx:]
            bin_all = self.concatchar_encoder.encode_spelling(w)
        elif opt=='REPLACE':
            rep_idx = np.random.randint(0, len(word))
            rep_char = np.random.choice([c for c in string.ascii_lowercase])
            w = word[:rep_idx] + rep_char + w[(rep_char+1):]
            bin_all = self.concatchar_encoder.encode_spelling(w)
        return np.repeat(np.array([bin_all]), 3, axis=0).reshape((1, len(self.concatchar_encoder)))[0]


    def jumble_char(self, word, opt):
        pass
