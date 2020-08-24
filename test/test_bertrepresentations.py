
import unittest

from simplerepresentations import RepresentationModel
import numpy as np

from shorttext.utils.transformers import WrappedBERTEncoder

class SimpleRepresentationsSentenceEncoder:
    def __init__(self):
        self.representation_model = RepresentationModel(
            model_type='bert',
            model_name='bert-base-uncased',
            batch_size=5,
            max_seq_length=48,
            combination_method='cat',
            last_hidden_to_use=4
        )

    def encode_sentences(self, sentences):
        all_sentences_representations, all_token_representations = \
            self.representation_model(text_a=sentences)
        return all_sentences_representations, all_token_representations


class TestRepresentations(unittest.TestCase):
    def test_representations(self):
        encoder = WrappedBERTEncoder()
        representer = SimpleRepresentationsSentenceEncoder()

        sentences = ['I love Python.',
                     'Topological quantum computing is interesting.',
                     'shorttext is a good package.',
                     'Thank you. You are welcome.',
                     'Liquid crystal is a good ground for testing statistical field theory.']

        sent_representations1, token_representations1, _ = encoder.encode_sentences(sentences)
        sent_representations2, token_representations2 = representer.encode_sentences(sentences)

        np.testing.assert_almost_equal(sent_representations1, sent_representations2)
        np.testing.assert_almost_equal(token_representations1, token_representations2)
