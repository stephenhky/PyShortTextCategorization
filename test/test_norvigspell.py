
import unittest
import sys

if sys.version_info[0]==2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

import shorttext

class TestSpellCheck(unittest.TestCase):
    def setUp(self):
        self.text = urlopen('http://norvig.com/big.txt').read()
        if sys.version_info[0]==3:
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