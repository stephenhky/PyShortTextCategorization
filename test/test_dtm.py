
import re

import pandas as pd
import pytest

import shorttext
from shorttext.utils import stemword, tokenize


def test_inaugural():
    # preparing data
    usprez = shorttext.data.inaugural()
    docids = sorted(usprez.keys())
    usprez = [' '.join(usprez[docid]) for docid in docids]
    usprezdf = pd.DataFrame({'yrprez': docids, 'speech': usprez})
    usprezdf = usprezdf[['yrprez', 'speech']]

    # preprocesser defined
    pipeline = [
        lambda s: re.sub(r'[^\w\s]', '', s),
        lambda s: re.sub(r'[0-9]', '', s),
        lambda s: s.lower(),
        lambda s: ' '.join([stemword(token) for token in tokenize(s)])
    ]
    txtpreprocessor = shorttext.utils.text_preprocessor(pipeline)

    # corpus making
    docids = list(usprezdf['yrprez'])
    corpus = [txtpreprocessor(speech) for speech in usprezdf['speech']]

    # making DTM
    dtm = shorttext.utils.NumpyDocumentTermMatrix(corpus, docids, tfidf=True)

    # check results
    assert dtm.get_token_occurences(stemword('change'))['2009-Obama'] == pytest.approx(0.0138)
    numdocs, numtokens = dtm.npdtm.shape
    assert numdocs == 56
    assert numtokens == 5256
    assert dtm.get_total_termfreq('government') == pytest.approx(0.27865372986738407)
