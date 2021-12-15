
import unittest

import shorttext

class TestTextPreprocessing(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testStandardPipeline(self):
        preprocessor = shorttext.utils.standard_text_preprocessor_1()
        self.assertEqual(preprocessor('I love you.'), 'love')
        self.assertEqual(preprocessor('Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(preprocessor('I do not think.'), 'think')

    def testStandPipelineDifferentStopwords(self):
        preprocessor = shorttext.utils.standard_text_preprocessor_2()
        self.assertEqual(preprocessor('I love you.'), 'love')
        self.assertEqual(preprocessor('Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(preprocessor('I do not think.'), 'not think')


if __name__ == '__main__':
    unittest.main()