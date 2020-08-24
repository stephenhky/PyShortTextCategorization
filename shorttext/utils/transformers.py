
# reference: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

import warnings

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BERTObject:
    def __init__(self, model=None, tokenizer=None, device='cpu'):
        """

        :param model:
        :param tokenizer:
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
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)\
                                .to(self.device)
        else:
            self.tokenizer = tokenizer.to(self.device)


class WrappedBERTEncoder(BERTObject):
    """

    """
    def __init__(self, model=None, tokenizer=None, device='cpu'):
        """

        :param model:
        :param tokenizer:
        """
        super(WrappedBERTEncoder, self).__init__(model=model, tokenizer=tokenizer, device=device)

    def encode_sentences(self, sentences, numpy=False):
        """

        :param sentences:
        :return:
        """
        input_ids = []
        tokenized_texts = []

        for sentence in sentences:
            marked_text = '[CLS]' + sentence + '[SEP]'

            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                truncation=True,
                max_length=48,
                pad_to_max_length=True,
                return_tensors='pt'
            )

            tokenized_texts.append(self.tokenizer.tokenize(marked_text))
            input_ids.append(encoded_dict['input_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        segments_id = torch.LongTensor(np.array(input_ids > 0))

        with torch.no_grad():
            _, sentences_embeddings, hidden_state = self.model(input_ids, segments_id)

        alllayers_token_embeddings = torch.stack(hidden_state, dim=0)
        alllayers_token_embeddings = alllayers_token_embeddings.permute(1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = alllayers_token_embeddings[:, :, 9:, :]   # we want last 4 layers only

        token_embeddings = torch.reshape(processed_embeddings, (len(sentences), 48, -1))

        if numpy:
            sentences_embeddings = sentences_embeddings.detach().numpy()
            token_embeddings = token_embeddings.detach().numpy()

        return sentences_embeddings, token_embeddings, tokenized_texts
