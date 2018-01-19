
import unittest
import re

import pandas as pd
import shorttext


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
                    lambda s: ' '.join(map(shorttext.utils.textpreprocessing.stem, shorttext.utils.tokenize(s)))
                    ]
        txtpreproceesor = shorttext.utils.text_preprocessor(pipeline)

        # corpus making
        docids = list(usprezdf['yrprez'])
        corpus = [txtpreproceesor(speech).split(' ') for speech in usprezdf['speech']]

        # making DTM
        dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=True)

        # check results
        self.assertEqual(len(dtm.dictionary), 5533)
        self.assertAlmostEqual(dtm.get_token_occurences(shorttext.utils.textpreprocessing.stem('change'))['2009-Obama'], 0.013832785291105156)
        numdocs, numtokens = dtm.dtm.shape
        self.assertEqual(numdocs, 56)
        self.assertEqual(numtokens, 5533)
        self.assertAlmostEqual(dtm.get_total_termfreq('government'), 0.27478750039568517)


if __name__ == '__main__':
    unittest.main()
