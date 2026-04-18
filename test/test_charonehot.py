
from urllib.request import urlopen

import shorttext


def test_onehot_bigtxt():
    chartovec_encoder = shorttext.generators.initialize_SentenceToCharVecEncoder(
        urlopen('http://norvig.com/big.txt'),
        encoding='utf-8'
    )
    assert len(chartovec_encoder.dictionary) == 93
    assert chartovec_encoder.signalchar == "\n"
