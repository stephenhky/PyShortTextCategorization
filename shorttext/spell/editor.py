
import numba as nb

# from . import edits1_comb


@nb.njit
def edits1_comb(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    returned_set = set(deletes + transposes + replaces + inserts)

    return returned_set


def compute_set_edits1(word):
    return edits1_comb(word)


def compute_set_edits2(word):
    return (e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))
