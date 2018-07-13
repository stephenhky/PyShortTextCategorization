
import unittest
import os

from shorttext.spell.sakaguchi import SCRNNSpellCorrector
from shorttext.smartload import smartload_compact_model

class TestSCRNN(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def generalproc(self, operation, typo='langudge', recommendation='language'):
        corrector = SCRNNSpellCorrector(operation)
        corrector.train('I am a nerd . Natural language processing is sosad .')
        corrector.save_compact_model('./sosad_'+operation+'_sakaguchi.bin')

        corrector2 = smartload_compact_model('./sosad_'+operation+'_sakaguchi.bin', None)
        self.assertEqual(corrector.correct(typo), corrector2.correct(typo))

        print('typo: '+typo+'  recommendation: '+corrector.correct(typo)+' ('+recommendation+')')

        os.remove('./sosad_'+operation+'_sakaguchi.bin')

    def test_NOISE_INSERT(self):
        self.generalproc('NOISE-INSERT')

    def test_NOISE_DELETE(self):
        self.generalproc('NOISE-DELETE')

    def test_NOISE_REPLACE(self):
        self.generalproc('NOISE-REPLACE', typo='procsesing', recommendation='processing')

    def test_JUMBLE_WHOLE(self):
        self.generalproc('JUMBLE-WHOLE')

    def test_JUMBLE_BEG(self):
        self.generalproc('JUMBLE-BEG')

    def test_JUMBLE_END(self):
        self.generalproc('JUMBLE-END')

    def test_JUMBLE_INT(self):
        self.generalproc('JUMBLE-INT')


if __name__ == '__main__':
    unittest.main()

