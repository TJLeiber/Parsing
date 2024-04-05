import re
from io import StringIO
import pandas as pd
import torch

def get_dm_adj_mtrx(df, seq_length):

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

def extract_from_dm(dm_examples_str: str):
    '''takes as input a set of dm examples as a single string and outputs
    a list of lists of strings, representing sentences (in decreasing order of their length)

    and their respective adjacency matrices as a tensor of size [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
    '''

    examples = re.split(r"\n#\d+", dm_examples_str)

    # get a list of all examples in string format (excluding the very first line since it is simply a title)
    tabular_strings = examples[1:]

    # list containing a dataframe for each example (write each df into a buffer file (tsv file) and call pd.read_csv on it)
    df_lst = [pd.read_csv(StringIO(tab_str), sep="\t", header=None) for tab_str in tabular_strings]

    # get the sentences
    sentences = [df[1].tolist() for df in df_lst]

    max_length = len(max(sentences, key=len))

    # get the labels (adjacency matrices) for each example
    Y = [get_dm_adj_mtrx(df, max_length) for df in df_lst]

    # concatenate all adjacency matrices along a new dimension
    Y = torch.stack(Y)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

    return sentences, Y

def get_vocab(sentences: list, vocab=None):
  '''returns a vocabulary given a list of sentences.
  if not otherwise specified
  '''

  vocab = ['<PAD>'] + ['<UNK>']
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

def get_sentence_lengths(sentences):
    '''expects as input sequences (vectorized or non-vectorized list of lists (sentences))
    and outputs a torch tensor of size len(sentences) indicating for the length of each sequence
    '''

    # get sentence lengths vector of size  BATCH_SIZE
    sent_lengths = torch.tensor(list(map(len, sentences)), dtype=torch.long)

    return sent_lengths


def pad_vect_sentences(vectorized_sents, w2i: dict):
  '''input: vectorized sent list (ordered descendingly), w2i mapping
     output: padded tensor of shape [BATCH_SIZE x SEQ_LENGTH] containing resp idx of each tok in a sequence
  '''

  sent_lengths = get_sentence_lengths(vectorized_sents)
  max_length = sent_lengths.max()

  # padded sequences tensor of size [BATCH_SIZE x SEQ_LENGTH]
  pad_idx = w2i["<PAD>"]
  sent_tensor = torch.full((len(vectorized_sents), max_length), pad_idx, dtype=torch.int32)

  for idx, (sent, sent_len) in enumerate(zip(vectorized_sents, sent_lengths)):
      sent_tensor[idx, :sent_len] = torch.tensor(sent, dtype=torch.long)

  # tensor of size [BATCH_SIZE x SEQ_LENGTH]
  # no need to sort in decreasing order since the examples were already sorted in the very beginning to allow mapping to Y_train, i.e. adjacency matrices
  X = sent_tensor

  return X
