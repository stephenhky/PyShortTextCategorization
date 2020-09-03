
from itertools import product

import numpy as np
import torch
from ...utils.transformers import WrappedBERTEncoder


class BERTScorer:
    """ This is the class that compute the BERTScores between sentences. BERTScores
    include recall BERTScores, precision BERTScores, and F1 BERTSscores.
    For more information, please refer to this paper:

    Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi,
    "BERTScore: Evaluating Text Generation with BERT," arXiv:1904.09675 (2019). [`arXiv
    <https://arxiv.org/abs/1904.09675>`_]

    """
    def __init__(
            self,
            model=None,
            tokenizer=None,
            max_length=48,
            nbencodinglayers=4,
            device='cpu'
    ):
        """ It is the class that compute the BERTScores between sentences.

        :param model: BERT model (default: None, with model `bert-base-uncase` to be used)
        :param tokenizer: BERT tokenizer (default: None, with model `bert-base-uncase` to be used)
        :param max_length: maximum number of tokens of each sentence (default: 48)
        :param nbencodinglayers: number of encoding layers (taking the last layers to encode the sentences, default: 4)
        :param device: device the language model is stored (default: `cpu`)
        :type model: str
        :type tokenizer: str
        :type max_length: int
        :type device: str
        """
        self.encoder = WrappedBERTEncoder(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            nbencodinglayers=nbencodinglayers,
            device=device)
        self.device = self.encoder.device
        self.cosine_fcn = torch.nn.CosineSimilarity(dim=0).to(self.device)

    def compute_matrix(self, sentence_a, sentence_b):
        """ Compute the table of similarities between all pairs of tokens. This is used
        for calculating the BERTScores.

        :param sentence_a: first sentence
        :param sentence_b: second sentence
        :return: similarity matrix of between tokens in two sentences
        :type sentence_a: str
        :type sentence_b: str
        :rtype: numpy.ndarray
        """
        cos = self.cosine_fcn
        _, sentence_a_tokens_embeddings, sentence_a_tokens = self.encoder.encode_sentences([sentence_a])
        _, sentence_b_tokens_embeddings, sentence_b_tokens = self.encoder.encode_sentences([sentence_b])

        similarity_matrix = torch.zeros((len(sentence_a_tokens[0])-2, len(sentence_b_tokens[0])-2),
                                        device=self.device)

        for i, j in product(range(len(sentence_a_tokens[0])-2), range(len(sentence_b_tokens[0])-2)):
            similarity_matrix[i, j] = cos(sentence_a_tokens_embeddings[0][i+1],
                                          sentence_b_tokens_embeddings[0][j+1])

        return similarity_matrix

    def recall_bertscore(self, reference_sentence, test_sentence):
        """ Compute the recall BERTScore between two sentences.

        :param reference_sentence: reference sentence
        :param test_sentence: test sentence
        :return: recall BERTScore between the two sentences
        :type reference_sentence: str
        :type test_sentence: str
        :rtype: float
        """
        similarity_matrix = self.compute_matrix(reference_sentence, test_sentence)
        recall = torch.mean(torch.max(similarity_matrix, axis=1).values)
        return np.float(recall.detach().numpy())

    def precision_bertscore(self, reference_sentence, test_sentence):
        """ Compute the precision BERTScore between two sentences.

        :param reference_sentence: reference sentence
        :param test_sentence: test sentence
        :return: precision BERTScore between the two sentences
        :type reference_sentence: str
        :type test_sentence: str
        :rtype: float
        """
        similarity_matrix = self.compute_matrix(reference_sentence, test_sentence)
        precision = torch.mean(torch.max(similarity_matrix, axis=0).values)
        return np.float(precision.detach().numpy())

    def f1score_bertscore(self, reference_sentence, test_sentence):
        """ Compute the F1 BERTScore between two sentences.

        :param reference_sentence: reference sentence
        :param test_sentence: test sentence
        :return: F1 BERTScore between the two sentences
        :type reference_sentence: str
        :type test_sentence: str
        :rtype: float
        """
        recall = self.recall_bertscore(reference_sentence, test_sentence)
        precision = self.precision_bertscore(reference_sentence, test_sentence)
        return 2*recall*precision/(recall+precision)
