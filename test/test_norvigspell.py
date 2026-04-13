
from urllib.request import urlopen

import shorttext


def test_norvig():
    text = urlopen('http://norvig.com/big.txt').read()
    text = text.decode("utf-8")
    speller = shorttext.spell.NorvigSpellCorrector()
    speller.train(text)

    assert speller.correct('apple') == 'apple'
    assert speller.correct('appl') == 'apply'
