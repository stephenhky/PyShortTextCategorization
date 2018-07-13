
from .edits1_comb import edits1_comb


def compute_set_edits1(word):
    return edits1_comb(word)


def compute_set_edits2(word):
    return (e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))
