# used to implement preprocessing of data building blocks

# used to implement preprocessing of data building blocks
import wget
import zipfile
import pandas as pd
import re
from io import StringIO
import torch
import torch.nn as nn

def load_data():
    '''downloads the data for semantic dependencies'''
    # download all data
    wget.download("http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/sdp_2015.zip")

    # extract zip file
    with zipfile.ZipFile("sdp_2015.zip", "r") as zip_ref:
        zip_ref.extractall("./")

    paths = ["./sdp_2015/en.dm.sdp.train",
    "./sdp_2015/en.dm.sdp.dev",
    "./sdp_2015/en.id.dm.sdp",
    "./sdp_2015/en.ood.dm.sdp"]

    data_dict = {"train": "", "dev": "", "test": {"ood": "", "id": ""}}

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith("train"):
                data_dict["train"] = f.read()
            elif path.endswith("dev"):
                data_dict["dev"] = f.read()
            elif path.startswith("end.id"):
                data_dict["test"]["id"] = f.read()
            else:
                data_dict["test"]["ood"] = f.read()

    return data_dict

def get_adj_mtrx(df, seq_length):

  # adjacency matrix as torch tensor of size [seq_length x seq_length]
  adj_mtrx = torch.zeros(seq_length, seq_length)

  # get root
  for idx, val in enumerate(df[4]):
    if val == '+':
      adj_mtrx[idx, idx] += 1

  # get head-dependent pairs
  nb_pred = 0
  for i, head in enumerate(df[5]):
    if head == '+':
      nb_pred += 1
      for j, dep_label in enumerate(df[6 + nb_pred]):
        if dep_label != '_':
          adj_mtrx[i, j] += 1

  return adj_mtrx

def preprocessing(examples_str):

    examples = re.split(r"\n#\d+", examples_str)

    # get a list of all examples in string format (excluding the very first line since it is simply a title)
    tabular_strings = examples[1:]

    # list containing a dataframe for each example (write each df into a buffer file (tsv file) and call pd.read_csv on it)
    df_lst = [pd.read_csv(StringIO(tab_str), sep="\t", header=None) for tab_str in tabular_strings]

    # sort the dataframes in decreasing order of their sentence lengths
    # will accelerate computation during biLSTM encoding
    df_lst.sort(key=len, reverse=True)

    # get the sentences
    sentences = [df[1].tolist() for df in df_lst]
    max_length = len(sentences[0])  # length of the longest sentence
    # get the labels (adjacency matrices) for each example
    Y = [get_adj_mtrx(df, max_length) for df in df_lst]

    # concatenate all adjacency matrices along a new dimension
    Y = torch.stack(Y)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

    # get the vocabulary
    vocab = ['<pad>'] + sorted(set(pd.concat([df[1] for df in df_lst])))

    # get quick access to indices for vectorization
    w2i = {word: idx for idx, word in enumerate(vocab)}

    # get example sentences as indices
    vectorized_sents = [[w2i[word] for word in sentence] for sentence in sentences]

    # padding
    # get sentence lengths vector of size  BATCH_SIZE
    sent_lengths = torch.tensor(list(map(len, vectorized_sents)), dtype=torch.int8)

    # padded sequences tensor of size [BATCH_SIZE x SEQ_LENGTH]
    sent_tensor = torch.zeros((len(vectorized_sents), max_length), dtype=torch.long)

    for idx, (sent, sent_len) in enumerate(zip(vectorized_sents, sent_lengths)):
        sent_tensor[idx, :sent_len] = torch.tensor(sent, dtype=torch.long)

    # tensor of size [BATCH_SIZE x SEQ_LENGTH]
    # no need to sort in decreasing order since the examples were already sorted in the very beginning to allow mapping to Y_train, i.e. adjacency matrices
    X = sent_tensor

    return X, Y
