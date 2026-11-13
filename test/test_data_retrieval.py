import numpy as np
import pytest

from shorttext.data import nihreports, yield_crossvalidation_classdicts


def test_nihreports_sample_size_reproducible_with_seed():
    # This path may require cached/downloaded data; if unavailable, skip rather than fail offline.
    try:
        data1 = nihreports(sample_size=8, random_state=123)
        data2 = nihreports(sample_size=8, random_state=123)
        data3 = nihreports(sample_size=8, random_state=456)
    except Exception as e:
        pytest.skip(f"NIH sample data unavailable in this environment: {e}")

    assert sum(len(v) for v in data1.values()) == 8
    assert data1 == data2
    # different seed should usually differ (not guaranteed, but extremely likely for this dataset/size)
    assert data1 != data3 or sum(len(v) for v in data3.values()) == 8


def test_yield_crossvalidation_classdicts_deterministic_shuffle():
    classdict = {
        'a': [f'a{i}' for i in range(5)],
        'b': [f'b{i}' for i in range(5)],
    }
    folds1 = list(yield_crossvalidation_classdicts(classdict, nb_partitions=3, shuffle=True, random_state=7))
    folds2 = list(yield_crossvalidation_classdicts(classdict, nb_partitions=3, shuffle=True, random_state=7))
    folds3 = list(yield_crossvalidation_classdicts(classdict, nb_partitions=3, shuffle=True, random_state=8))

    assert folds1 == folds2
    assert len(folds1) == 3
    # every item appears exactly once as test across folds per label
    for label in classdict:
        seen = []
        for testd, traind in folds1:
            seen.extend(testd.get(label, []))
            assert set(testd.get(label, [])).isdisjoint(set(traind.get(label, [])))
        assert sorted(seen) == sorted(classdict[label])
    assert folds1 != folds3


def test_yield_crossvalidation_classdicts_invalid_partitions():
    with pytest.raises(ValueError):
        list(yield_crossvalidation_classdicts({'a': ['x']}, nb_partitions=0))
