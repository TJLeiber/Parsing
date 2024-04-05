import torch
import torch.nn as nn
import gensim.downloader as api
import json

class GloVe:
    def __init__(self):
        self.model = api.load("glove-wiki-gigaword-100")

        # get a tensor version of the embeddings as attribute and unclude "<UNK>" and "<PAD>" tokens
        self.weights = torch.FloatTensor(self.model.vectors)
        pad_unk = torch.randn(2, self.weights.shape[1]) * 0.01
        self.weights = torch.cat((self.weights, pad_unk), dim=0)
        self.embed_dim = self.weights.shape[1]

        # assign the respective indices
        self.w2i = {token: token_index for token_index, token in enumerate(self.model.index_to_key)}
        self.w2i["<PAD>"] = len(self.w2i)
        self.w2i["<UNK>"] = len(self.w2i)
        self.vocab = [tok for tok in self.w2i]

    def serialize(self):
        self.model.save("glove-wiki-gigaword-100.model")
        torch.save(self.weights, "./glove.pt")
        with open("./w2i.json", "w") as f:
          json.dump(self.w2i, f)

    def load(self):
      glove = torch.load("./glove.pt")
      return glove# Everything Embedding code related goes here
