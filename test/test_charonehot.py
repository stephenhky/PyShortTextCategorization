
import unittest

import shorttext
import urllib2

class TestCharOneHot(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBigTxt(self):
        chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(urllib2.urlopen('http://norvig.com/big.txt', 'r'))
        self.assertEqual(0, len(chartovec_encoder.dictionary))
        self.assertEqual('\n', chartovec_encoder.signalchar)


if __name__ == '__main__':
    unittest.main()
