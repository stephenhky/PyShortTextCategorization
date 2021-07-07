
import unittest
from urllib.request import urlopen

import shorttext


class TestCharOneHot(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_BigTxt(self):
        chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(urlopen('http://norvig.com/big.txt'),
                                                                              encoding='utf-8')
        self.assertEqual(93, len(chartovec_encoder.dictionary))
        self.assertEqual('\n', chartovec_encoder.signalchar)


if __name__ == '__main__':
    unittest.main()
