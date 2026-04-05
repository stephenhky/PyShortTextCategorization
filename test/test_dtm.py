
import pytest

import shorttext
from shorttext.utils import stemword
from shorttext.utils.textpreprocessing import standard_text_preprocessor_1


def test_inaugural():
    # preparing data
    usprez = shorttext.data.inaugural()
    docids = sorted(usprez.keys())
    usprez = [' '.join(usprez[docid]) for docid in docids]

    # preprocesser defined
    txtpreprocessor = standard_text_preprocessor_1()

    # corpus making
    corpus = [txtpreprocessor(speech) for speech in usprez]

    # making DTM
    dtm = shorttext.utils.NumpyDocumentTermMatrix(corpus, docids, tfidf=True)

    # check results
    assert dtm.get_token_occurences(stemword('change'))['2009-Obama'] == pytest.approx(0.9400072584914713)
    assert dtm.nbdocs == 56
    assert dtm.nbtokens == 5075
    assert dtm.get_total_termfreq(stemword('government')) == pytest.approx(37.82606692473982)
