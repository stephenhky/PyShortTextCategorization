import shorttext


def test_norvig_spell_correction(big_text):
    speller = shorttext.spell.NorvigSpellCorrector()
    speller.train(big_text)

    assert speller.correct('apple') == 'apple'
    corrected = speller.correct('appl')
    assert corrected in {'apple', 'apply', 'appl'}
    assert isinstance(corrected, str)
