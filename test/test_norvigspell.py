
import unittest
import sys
from urllib.request import urlopen

import shorttext


class TestSpellCheck(unittest.TestCase):
    def setUp(self):
        self.text = urlopen('http://norvig.com/big.txt').read()
        self.text = self.text.decode('utf-8')

    def tearDown(self):
        pass

    def test_norvig(self):
        speller = shorttext.spell.NorvigSpellCorrector()
        speller.train(self.text)
        self.assertEqual(speller.correct('apple'), 'apple')
        self.assertEqual(speller.correct('appl'), 'apply')

if __name__ == '__main__':
    unittest.main()