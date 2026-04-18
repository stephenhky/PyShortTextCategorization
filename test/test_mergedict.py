
from functools import reduce

import pytest

from shorttext.data.data_retrieval import mergedict


def test_mergedict_1():
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    dict2 = {'a': 4, 'd': 5, 'e': 6}
    dict3 = {'f': 7, 'g': 8, 'h': 9}

    old_merged_dict = mergedict([dict1, dict2, dict3])
    new_merged_dict = reduce(lambda x, y: x | y, [dict1, dict2, dict3])

    assert len(old_merged_dict) == len(new_merged_dict)
    for key, value in old_merged_dict.items():
        assert value == new_merged_dict[key]
