
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
        self.assertEqual(len(dtm.dictionary), 5253)   # from spacy tokenization to space-delimiation, the number of tokens changed from 5252 to 5253
        self.assertAlmostEqual(dtm.get_token_occurences(stemmer.stemWord('change'))['2009-Obama'], 0.01387942827805605)
        numdocs, numtokens = dtm.dtm.shape
        self.assertEqual(numdocs, 56)
        self.assertEqual(numtokens, 5253)
        self.assertAlmostEqual(dtm.get_total_termfreq('government'), 0.27872964951168006)


if __name__ == '__main__':
    unittest.main()
