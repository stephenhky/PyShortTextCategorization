import unittest

class SampleTest(unittest.TestCase):
    def setUp(self):
        self.sample_var = True

    def testSampleTestCase(self):
        self.assertEqual(True, self.sample_var)

if __name__ == '__main__':
    unittest.main()