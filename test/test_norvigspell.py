
import unittest
import sys

if sys.version_info[0]==2:
    import urllib2
else:
    import urllib.request as urllib2

import shorttext

class TestSpellCheck(unittest.TestCase):
    def setUp(self):
        self.text = urllib2.urlopen('https://norvig.com/big.txt').read()

    def tearDown(self):
        pass

    def test_norvig(self):
        speller = shorttext.spell.NorvigSpellCorrector()
        speller.train(self.text)
        self.assertEqual(speller.correct('apple'), 'apple')
        self.assertEqual(speller.correct('appl'), 'apply')

if __name__ == '__main__':
    unittest.main()