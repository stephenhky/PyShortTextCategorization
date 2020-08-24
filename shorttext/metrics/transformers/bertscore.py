
from itertools import product

import numpy as np
import torch
from ...utils.transformers import WrappedBERTEncoder


class BERTScorer:
    def __init__(self, model=None, tokenizer=None):
        """

        :param model:
        :param tokenizer:
        """
        self.encoder = WrappedBERTEncoder(model=model, tokenizer=tokenizer)
        self.cosine_fcn = torch.nn.CosineSimilarity(dim=0)

    def compute_matrix(self, sentence_a, sentence_b):
        cos = self.cosine_fcn
        _, sentence_a_tokens_embeddings, sentence_a_tokens = self.encoder.encode_sentences([sentence_a])
        _, sentence_b_tokens_embeddings, sentence_b_tokens = self.encoder.encode_sentences([sentence_b])

        similarity_matrix = torch.zeros((len(sentence_a_tokens)-2, len(sentence_b_tokens)-2))

        for i, j in product(range(len(sentence_a_tokens)-2), range(len(sentence_b_tokens)-2)):
            similarity_matrix[i, j] = cos(sentence_a_tokens_embeddings[0][i+1],
                                          sentence_b_tokens_embeddings[0][j+1])

        return similarity_matrix

    def recall_bertscore(self, reference_sentence, test_sentence):
        similarity_matrix = self.compute_matrix(reference_sentence, test_sentence)
        return np.float(torch.mean(torch.max(similarity_matrix, axis=1).values))

    def precision_bertscore(self, reference_sentence, test_sentence):
        similarity_matrix = self.compute_matrix(reference_sentence, test_sentence)
        return np.float(torch.mean(torch.max(similarity_matrix, axis=0).values))

    def f1score_bertscore(self, reference_sentence, test_sentence):
        recall = self.recall_bertscore(reference_sentence, test_sentence)
        precision = self.precision_bertscore(reference_sentence, test_sentence)
        return 2*recall*precision/(recall+precision)
