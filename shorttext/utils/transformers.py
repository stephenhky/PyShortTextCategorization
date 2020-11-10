
# reference: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

import warnings

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BERTObject:
    """ The base class for BERT model that contains the embedding model and the tokenizer.

    For more information, please refer to the following paper:

    Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805 (2018). [`arXiv
    <https://arxiv.org/abs/1810.04805>`_]

    """
    def __init__(self, model=None, tokenizer=None, device='cpu'):
        """ The base class for BERT model that contains the embedding model and the tokenizer.

        :param model: BERT model (default: None, with model `bert-base-uncase` to be used)
        :param tokenizer: BERT tokenizer (default: None, with model `bert-base-uncase` to be used)
        :param device: device the language model is stored (default: `cpu`)
        :type model: str
        :type tokenizer: str
        :type device: str
        """
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                warnings.warn("CUDA is not available. Device set to 'cpu'.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                                   output_hidden_states=True)\
                            .to(self.device)
        else:
            self.model = model.to(self.device)

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

        self.number_hidden_layers = self.model.config.num_hidden_layers


class WrappedBERTEncoder(BERTObject):
    """ This is the class that encodes sentences with BERT models.

    For more information, please refer to the following paper:

    Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805 (2018). [`arXiv
    <https://arxiv.org/abs/1810.04805>`_]

    """
    def __init__(
            self,
            model=None,
            tokenizer=None,
            max_length=48,
            nbencodinglayers=4,
            device='cpu'
    ):
        """ This is the constructor of the class that encodes sentences with BERT models.

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
        super(WrappedBERTEncoder, self).__init__(model=model, tokenizer=tokenizer, device=device)
        self.max_length = max_length
        self.nbencodinglayers = nbencodinglayers

    def encode_sentences(self, sentences, numpy=False):
        """ Encode the sentences into numerical vectors, given by a list of strings.

        It can output either torch tensors or numpy arrays.

        :param sentences: list of strings to encode
        :param numpy: output a numpy array if `True`; otherwise, output a torch tensor. (Default: `False`)
        :return: encoded vectors for the sentences
        :type sentences: list
        :type numpy: bool
        :rtype: numpy.array or torch.Tensor
        """
        input_ids = []
        tokenized_texts = []

        for sentence in sentences:
            marked_text = '[CLS]' + sentence + '[SEP]'

            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                pad_to_max_length=True,
                return_tensors='pt'
            )

            tokenized_texts.append(self.tokenizer.tokenize(marked_text))
            input_ids.append(encoded_dict['input_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        segments_id = torch.LongTensor(np.array(input_ids > 0))
        input_ids = input_ids.to(self.device)
        segments_id = segments_id.to(self.device)

        with torch.no_grad():
            _, sentences_embeddings, hidden_state = self.model(input_ids, segments_id)

        alllayers_token_embeddings = torch.stack(hidden_state, dim=0)
        alllayers_token_embeddings = alllayers_token_embeddings.permute(1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = alllayers_token_embeddings[:, :, (self.number_hidden_layers+1-self.nbencodinglayers):, :]

        token_embeddings = torch.reshape(processed_embeddings, (len(sentences), self.max_length, -1))

        if numpy:
            sentences_embeddings = sentences_embeddings.detach().numpy()
            token_embeddings = token_embeddings.detach().numpy()

        return sentences_embeddings, token_embeddings, tokenized_texts
