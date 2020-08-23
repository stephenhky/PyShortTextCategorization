
# reference: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class WrappedBERTEncoder:
    def __init__(self, model=None, tokenizer=None):
        if model is None:
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                                   output_hidden_state=True)
        else:
            self.model = model

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

    def encode_sentences(self, sentences):
        input_ids = []
        # attention_masks = []
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

        token_embeddings = torch.stack(hidden_state, dim=0)
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = token_embeddings[:, :, 9:, :]   # we want last 4 layers only

        embeddings = torch.reshape(processed_embeddings, (len(sentences), 48, -1))
        embeddings = embeddings.detach().numpy()

        return sentences_embeddings, embeddings
