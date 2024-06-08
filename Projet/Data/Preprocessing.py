import re
from io import StringIO
import pandas as pd
import torch

def get_dm_adj_mtrx(df, seq_length, include_root=True):

  # adjacency matrix as torch tensor of size [seq_length x seq_length]
  adj_mtrx = torch.zeros(seq_length, seq_length)

  # get root
  if include_root:
    for idx, val in enumerate(df[4]):
      if val == '+':
        adj_mtrx[0, idx + 1] += 1 # first word in a seq is always '<ROOT>'

  # get head-dependent pairs
  nb_pred = 0
  for i, head in enumerate(df[5]):
    if head == '+':
      nb_pred += 1
      for j, dep_label in enumerate(df[6 + nb_pred]):
        if dep_label != '_':
          if include_root:
            adj_mtrx[i + 1, j + 1] += 1 # i + 1 and j + 1, because root is not included in df, but will be in adjacency matrix
          else:
            adj_mtrx[i, j] += 1 # if we do not predict root

  return adj_mtrx

def extract_from_dm(dm_examples_str: str, max_length=None, include_root=True):
    '''takes as input a set of dm examples as a single string and outputs
    a list of lists of strings, representing sentences (in decreasing order of their length)

    and their respective adjacency matrices as a tensor of size [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
    dm_examples_str: string containing dm organised data
    max_length: inclusive upper bound for sentence length of sentences to be added to be returned (if no argument is given no upper bound will be considered)
    '''

    examples = re.split(r"\n#\d+", dm_examples_str)

    # get a list of all examples in string format (excluding the very first line since it is simply a title)
    tabular_strings = examples[1:]

    # list containing a dataframe for each example (write each df into a buffer file (tsv file) and call pd.read_csv on it)
    df_lst = [pd.read_csv(StringIO(tab_str), sep="\t", header=None) for tab_str in tabular_strings]

    # get the sentences
    if max_length is not None:
      sentences = [["<ROOT>"] + df[1].tolist() for df in df_lst if len(df[1].tolist()) <= max_length]
      actual_max_len = len(max(sentences, key=len)) - 1 # root is excluded
    else: # i.e. if no upper bound was given
      sentences = [["<ROOT>"] + df[1].tolist() for df in df_lst]
      actual_max_len = len(max(sentences, key=len)) # maximum sentence length in data can be less than upper bound (max_length)

    if include_root:
      actual_max_len = len(max(sentences, key=len)) # root is excluded
    else:
      actual_max_len = len(max(sentences, key=len)) - 1 # root is excluded

    # get the labels (adjacency matrices) for each example

    # only create adjacency matrices for sequences <= max_length
    if max_length is not None:
      if include_root:
        Y = [get_dm_adj_mtrx(df, actual_max_len) for df in df_lst if len(df[1].tolist()) <= max_length]
      else: # if we do not want to include root
        Y = [get_dm_adj_mtrx(df, actual_max_len, include_root=False) for df in df_lst if len(df[1].tolist()) <= max_length]

    # create adjacency matrices for sequences of any length
    else:
      if include_root:
        Y = [get_dm_adj_mtrx(df, actual_max_len) for df in df_lst]
      else: # if we do not want to include root 
        Y = [get_dm_adj_mtrx(df, actual_max_len, include_root=False) for df in df_lst]


    # concatenate all adjacency matrices along a new dimension
    Y = torch.stack(Y)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

    return sentences, Y

def get_vocab(sentences: list, vocab=None):
  '''returns a vocabulary given a list of sentences.
  if not otherwise specified
  '''

  vocab = ['<PAD>'] + ['<UNK>'] + ["<ROOT>"] + ['<DROP>']
  additions = []
  for sentence in sentences:
    for word in sentence:
      additions.append(word)

  add = sorted(set(additions))

  vocab = vocab + add

  return vocab

def get_w2i(vocab: list):
  '''given a vocab (index to word mapping) return the word to index mapping'''

  w2i = {word: idx for idx, word in enumerate(vocab)}

  return w2i

def vectorize(sentences: list, w2i: dict):
  '''vectorize a list of lists relative to some vocabulary to index mapping'''

  # get example sentences as indices
  vectorized_sents = [[w2i[word] if word in w2i else w2i["<UNK>"] for word in sentence] for sentence in sentences]

  return vectorized_sents

def get_sentence_lengths(sentences, include_root=True):
    '''expects as input sequences (vectorized or non-vectorized list of lists (sentences))
    and outputs a torch tensor of size len(sentences) indicating for the length of each sequence
    '''

    # get sentence lengths vector of size  BATCH_SIZE
    if include_root:
      sent_lengths = torch.tensor(list(map(len, sentences)), dtype=torch.long)
    else:
      sent_lengths = torch.tensor(list(map(lambda x: len(x) - 1, sentences)), dtype=torch.long)

    return sent_lengths


def pad_vect_sentences(vectorized_sents, w2i: dict):
  '''input: vectorized sent list (ordered descendingly), w2i mapping
     output: padded tensor of shape [BATCH_SIZE x SEQ_LENGTH] containing resp idx of each tok in a sequence
  '''

  sent_lengths = get_sentence_lengths(vectorized_sents)
  max_length = sent_lengths.max()

  # padded sequences tensor of size [BATCH_SIZE x SEQ_LENGTH]
  pad_idx = w2i["<PAD>"]
  sent_tensor = torch.full((len(vectorized_sents), max_length), pad_idx, dtype=torch.long)

  for idx, (sent, sent_len) in enumerate(zip(vectorized_sents, sent_lengths)):
      sent_tensor[idx, :sent_len] = torch.tensor(sent, dtype=torch.long)

  # tensor of size [BATCH_SIZE x SEQ_LENGTH]
  # no need to sort in decreasing order since the examples were already sorted in the very beginning to allow mapping to Y_train, i.e. adjacency matrices
  X = sent_tensor

  return X
