
import re
import string
from functools import reduce

import numpy as np
from shorttext.generators.charbase.char2vec import initSentenceToCharVecEncoder
from shorttext.utils.classification_exceptions import OperationNotDefinedException


default_alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
# NB. # is <eos>, _ is <unk>, @ is number
default_specialsignals = {'eos': '#', 'unk': '_', 'number': '@'}
default_signaldenotions = {'<eos>': 'eos', '<unk>': 'unk'}


class SpellingToConcatCharVecEncoder:
    def __init__(self, alph):
        self.charevec_encoder = initSentenceToCharVecEncoder(alph)

    def encode_spelling(self, spelling):
        spmat = self.charevec_encoder.encode_sentence(spelling, len(spelling))
        return spmat.sum(axis=0)

    def __len__(self):
        return len(self.charevec_encoder)


def hasnum(word):
    return len(re.findall('\\d', word)) > 0


class SCRNNBinarizer:
    """ A class used by Sakaguchi's spell corrector to convert text into numerical vectors.

    No documentation for this class.

    """
    def __init__(self, alpha, signalchar_dict):
        self.signalchar_dict = signalchar_dict
        self.concatchar_encoder = SpellingToConcatCharVecEncoder(alpha)
        self.char_dict = self.concatchar_encoder.charevec_encoder.dictionary

    def noise_char(self, word, opt, unchanged=False):
        bin_all = np.zeros((len(self.concatchar_encoder), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_all[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
        elif hasnum(word):
            bin_all[self.char_dict.token2id[default_specialsignals['number']]] += 1
        elif unchanged:
            bin_all = self.concatchar_encoder.encode_spelling(w).transpose()
        elif opt=='DELETE':
            if len(word) > 1:
                idx = np.random.randint(0, len(word))
                w = word[:idx] + word[(idx+1):]
            else:
                w = word
            bin_all = self.concatchar_encoder.encode_spelling(w).transpose()
        elif opt=='INSERT':
            ins_idx = np.random.randint(0, len(word)+1)
            ins_char = np.random.choice([c for c in string.ascii_lowercase])
            w = word[:ins_idx] + ins_char + word[ins_idx:]
            bin_all = self.concatchar_encoder.encode_spelling(w).transpose()
        elif opt=='REPLACE':
            rep_idx = np.random.randint(0, len(word))
            rep_char = np.random.choice([c for c in string.ascii_lowercase])
            w = word[:rep_idx] + rep_char + w[(rep_idx+1):]
            bin_all = self.concatchar_encoder.encode_spelling(w).transpose()
        else:
            raise OperationNotDefinedException('NOISE-'+opt)
        return np.array([ np.repeat(np.array([bin_all]), 3, axis=0).reshape((1, len(self.concatchar_encoder)*3))[0] ]).transpose(), w

    def jumble_char(self, word, opt, unchanged=False):
        if opt=='WHOLE':
            return self.jumble_char_whole(word, unchanged=unchanged)
        elif opt=='BEG':
            return self.jumble_char_beg(word, unchanged=unchanged)
        elif opt=='END':
            return self.jumble_char_end(word, unchanged=unchanged)
        elif opt=='INT':
            return self.jumble_char_int(word, unchanged=unchanged)
        else:
            raise OperationNotDefinedException('JUMBLE-'+opt)

    def jumble_char_whole(self, word, unchanged=False):
        bin_all = np.zeros((len(self.concatchar_encoder), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_all[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
        elif hasnum(word):
            bin_all[self.char_dict.token2id[default_specialsignals['number']]] += 1
        else:
            w = ''.join(np.random.choice([c for c in word], len(word), replace=False)) if not unchanged else word
            bin_all = self.concatchar_encoder.encode_spelling(w).transpose()
        bin_filler = np.zeros((len(self.concatchar_encoder)*2, 1))
        return np.concatenate((bin_all, bin_filler), axis=0), w

    def jumble_char_beg(self, word, unchanged=False):
        bin_initial = np.zeros((len(self.concatchar_encoder), 1))
        bin_end = np.zeros((len(self.concatchar_encoder), 1))
        bin_filler = np.zeros((len(self.concatchar_encoder), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_initial[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
        elif hasnum(word):
            bin_initial[self.char_dict.token2id[default_specialsignals['number']]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals['number']]] += 1
        else:
            w_init = ''.join(np.random.choice([c for c in word[:-1]], len(word)-1)) if not unchanged and len(w)>3 else word[:-1]
            w = w_init + word[-1]
            if len(w_init) > 0:
                bin_initial = self.concatchar_encoder.encode_spelling(w_init).transpose()
            bin_end = self.concatchar_encoder.encode_spelling(word[-1]).transpose()
        return reduce(lambda a, b: np.concatenate((a, b), axis=0), [bin_initial, bin_end, bin_filler]), w

    def jumble_char_end(self, word, unchanged=False):
        bin_initial = np.zeros((len(self.concatchar_encoder), 1))
        bin_end = np.zeros((len(self.concatchar_encoder), 1))
        bin_filler = np.zeros((len(self.concatchar_encoder), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_initial[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
        elif hasnum(word):
            bin_initial[self.char_dict.token2id[default_specialsignals['number']]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals['number']]] += 1
        else:
            w_end = ''.join(np.random.choice([c for c in word[1:]], len(word)-1)) if not unchanged and len(w)>3 else word[1:]
            w = word[0] + w_end
            bin_initial = self.concatchar_encoder.encode_spelling(word[0]).transpose()
            if len(w_end) > 0:
                bin_end = self.concatchar_encoder.encode_spelling(w_end).transpose()
        return reduce(lambda a, b: np.concatenate((a, b), axis=0), [bin_initial, bin_end, bin_filler]), w

    def jumble_char_int(self, word, unchanged=False):
        bin_initial = np.zeros((len(self.concatchar_encoder), 1))
        bin_middle = np.zeros((len(self.concatchar_encoder), 1))
        bin_end = np.zeros((len(self.concatchar_encoder), 1))
        w = word
        if word in default_signaldenotions.keys():
            bin_initial[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
            bin_middle[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals[default_signaldenotions[word]]]] += 1
        elif hasnum(word):
            bin_initial[self.char_dict.token2id[default_specialsignals['number']]] += 1
            bin_middle[self.char_dict.token2id[default_specialsignals['number']]] += 1
            bin_end[self.char_dict.token2id[default_specialsignals['number']]] += 1
        else:
            w_mid = ''.join(np.random.choice([c for c in word[1:-1]], len(word)-2)) if not unchanged and len(w)>3 else w[1:-1]
            w = word[0] + w_mid + word[-1]
            bin_initial = self.concatchar_encoder.encode_spelling(word[0]).transpose()
            if len(w_mid)>0:
                bin_middle = self.concatchar_encoder.encode_spelling(w_mid).transpose()
            bin_end = self.concatchar_encoder.encode_spelling(word[-1]).transpose()
        return reduce(lambda a, b: np.append(a, b, axis=0), [bin_initial, bin_middle, bin_end]), w

    def change_nothing(self, word, operation):
        if operation.upper().startswith('NOISE'):
            return self.noise_char(word, operation[6:], unchanged=True)
        else:
            return self.jumble_char(word, operation[7:], unchanged=True)

