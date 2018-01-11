
import numpy as np
from gensim.corpora import Dictionary
from scipy.sparse import dok_matrix
import pandas as pd


class DocumentTermMatrix:
    def __init__(self, corpus, docids=None):
        if docids == None:
            self.docid_dict = {i: i for i in range(len(corpus))}
            self.docids = range(len(corpus))
        else:
            if len(docids) == len(corpus):
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docids = docids
            elif len(docids) > len(corpus):
                self.docid_dict = {docid: i for i, docid in zip(range(len(corpus)), docids[:len(corpus)])}
                self.docids = docids[:len(corpus)]
            else:
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docid_dict = {i: i for i in range(len(docids), range(corpus))}
                self.docids = docids + range(len(docids), range(corpus))

        self.generate_dtm(corpus)

    def generate_dtm(self, corpus):
        self.dictionary = Dictionary(corpus)
        self.dtm = dok_matrix((len(corpus), len(self.dictionary)), dtype=np.int)
        for docid in self.docids:
            for tokenid, count in self.dictionary.doc2bow(corpus[self.docid_dict[docid]]):
                self.dtm[self.docid_dict[docid], tokenid] = count

    def get_termfreq(self, docid, token):
        return self.dtm[self.docid_dict[docid], self.dictionary.token2id[token]]

    def get_total_termfreq(self, token):
        return sum(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_doc_frequency(self, token):
        return len(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_token_occurences(self, token):
        return {self.docid_dict[docidx]: count for (docidx, _), count in self.dtm[:, token].items()}

    def get_doc_tokens(self, docid):
        return {self.dictionary.token2id[tokenid]: count for (_, tokenid), count in self.dtm[self.docid_dict[docid], :].items()}

    def generate_dtm_dataframe(self):
        tbl = pd.DataFrame(self.dtm.toarray())
        tbl.index = self.docids
        tbl.columns = map(lambda i: self.dictionary[i], range(len(self.dictionary)))
        return tbl
