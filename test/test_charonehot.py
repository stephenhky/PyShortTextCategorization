
import unittest
import sys

import shorttext
if sys.version_info[0]==2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

class TestCharOneHot(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_BigTxt(self):
        encoding = 'utf-8' if sys.version_info[0]==3 else None
        chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(urlopen('http://norvig.com/big.txt'),
                                                                              encoding=encoding)
        self.assertEqual(93, len(chartovec_encoder.dictionary))
        self.assertEqual('\n', chartovec_encoder.signalchar)


if __name__ == '__main__':
    unittest.main()
