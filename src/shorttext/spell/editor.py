
from typing import Generator

import numba as nb


@nb.njit
def compute_set_edits1(word: str) -> set[str]:
    """Generate all single-edit distance words.

    Creates all possible words that are one edit (insert, delete,
    transpose, replace) away from the input word.

    Args:
        word: Input word.

    Returns:
        Set of all possible single-edit variations.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    returned_set = set(deletes + transposes + replaces + inserts)

    return returned_set


@nb.njit
def compute_set_edits2(word: str) -> Generator[str, None, None]:
    """Generate all double-edit distance words.

    Creates all possible words that are two edits away from the
    input word by applying compute_set_edits1 to each result.

    Args:
        word: Input word.

    Yields:
        All possible double-edit variations.
    """
    return (e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))
