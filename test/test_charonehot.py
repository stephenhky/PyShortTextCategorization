import io

import shorttext


def test_onehot_encoder_from_text(big_text):
    chartovec_encoder = shorttext.generators.SentenceToCharVecEncoder.from_pretrained(
        io.StringIO(big_text)
    )
    assert len(chartovec_encoder.dictionary) > 0
    assert chartovec_encoder.signalchar == "\n"
    vec = chartovec_encoder.encode_sentence("apple", maxlen=10, startsig=False, endsig=False)
    assert vec.shape[0] == 10
    assert vec.shape[1] == len(chartovec_encoder.dictionary)
