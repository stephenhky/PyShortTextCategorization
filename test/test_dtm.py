import pytest

import shorttext
from shorttext.utils import stemword
from shorttext.utils.textpreprocessing import standard_text_preprocessor_1


def test_inaugural_dtm_tfidf():
    usprez = shorttext.data.inaugural()
    docids = sorted(usprez.keys())
    docs = [' '.join(usprez[docid]) for docid in docids]

    txtpreprocessor = standard_text_preprocessor_1()
    corpus = [txtpreprocessor(speech) for speech in docs]
    dtm = shorttext.utils.NumpyDocumentTermMatrix(corpus, docids, tfidf=True)

    # structural/deterministic properties
    assert dtm.nbdocs == 56
    assert dtm.nbtokens > 0
    assert '2009-Obama' in dtm.get_token_occurences(stemword('change'))
    assert dtm.get_token_occurences(stemword('change'))['2009-Obama'] > 0
    assert dtm.get_total_termfreq(stemword('government')) > 0

    # regression values for this environment/data snapshot (stable given same corpus+preprocessor)
    assert dtm.nbtokens == 5077
    assert dtm.get_token_occurences(stemword('change'))['2009-Obama'] == pytest.approx(0.9400072584914713)
    assert dtm.get_total_termfreq(stemword('government')) == pytest.approx(37.82606692473982)
