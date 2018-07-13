
def edits1_comb(str word):
    cdef str letters = 'abcdefghijklmnopqrstuvwxyz'
    cdef int i
    cdef list splits, deletes, transposes, replaces, inserts
    cdef set returned_set

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    returned_set = set(deletes + transposes + replaces + inserts)

    return returned_set

