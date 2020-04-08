
import unittest
import re

import pandas as pd
import shorttext
from shorttext.utils import stemword, tokenize


class TestDTM(unittest.TestCase):
    def test_inaugural(self):
        # preparing data
        usprez = shorttext.data.inaugural()
        docids = sorted(usprez.keys())
        usprez = [' '.join(usprez[docid]) for docid in docids]
        usprezdf = pd.DataFrame({'yrprez': docids, 'speech': usprez})
        usprezdf = usprezdf[['yrprez', 'speech']]

        # preprocesser defined
        pipeline = [lambda s: re.sub('[^\w\s]', '', s),
                    lambda s: re.sub('[\d]', '', s),
                    lambda s: s.lower(),
                    lambda s: ' '.join([stemword(token) for token in tokenize(s)])
                    ]
        txtpreprocessor = shorttext.utils.text_preprocessor(pipeline)

        # corpus making
        docids = list(usprezdf['yrprez'])
        corpus = [txtpreprocessor(speech).split(' ') for speech in usprezdf['speech']]

        # making DTM
        dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=True)

        # check results
        self.assertEqual(len(dtm.dictionary), 5406)
        self.assertAlmostEqual(dtm.get_token_occurences(stemword('change'))['2009-Obama'], 0.013801565936022027,
                               places=4)
        numdocs, numtokens = dtm.dtm.shape
        self.assertEqual(numdocs, 56)
        self.assertEqual(numtokens, 5406)
        self.assertAlmostEqual(dtm.get_total_termfreq('government'), 0.27584786568258396,
                               places=4)


if __name__ == '__main__':
    unittest.main()
