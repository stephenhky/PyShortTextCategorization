
import shorttext


def test_textpreprocessing_standard_pipeline():
    preprocessor = shorttext.utils.standard_text_preprocessor_1()
    assert preprocessor('I love you.') == 'love'
    assert preprocessor('Natural language processing and text mining on fire.') == 'natur languag process text mine fire'
    assert preprocessor('I do not think.') == 'think'

def test_textpreprocessing_standard_pipeline_stopwords():
    preprocessor = shorttext.utils.standard_text_preprocessor_2()
    assert preprocessor('I love you.') == 'love'
    assert preprocessor('Natural language processing and text mining on fire.') == 'natur languag process text mine fire'
    assert preprocessor('I do not think.') == 'not think'
