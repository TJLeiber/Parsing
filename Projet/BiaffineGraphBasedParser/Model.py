import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init


# CONVENTIONS
# SEQ_LENGTH (or SEQUENCE_LENGTH) is the uniform length of (padded) sentences in a batch


# ------------------------------------------------------------------- BERT ENCODING AND SUBWORD TOKEN PROBLEM --------------------------------------------------------------
# 3 functions which deal with converting lists of lists of sentences wplit into words into tensors containing contextual BERT representations of a sentence
# Subtokenization is dealt with by retaining the first subtoken of a word, 
# disgregarding the other subtokens of a word and then contracting the sequence

def contract_single_sequence(sequence, word_ids, max_length):
    '''
    sequence: torch.tensor of shape [1, UNCONTRACTED_MAX_LENGTH, 768]
    word_ids: list of length of tokenized sequence
    max_length: length of each padded sequence before tokenization c.f. contract_batch and get_and_contract_BERT functions
    '''

    # note that the number of unique word_ids is already given by max_length, i.e. the length of every padded sequence in the batch before tokenization

    with torch.no_grad():

      if len(word_ids) == max_length: # case of no sub word tokenization in a sequence
          return sequence # return a tensor of shape [1 X SEQ_LENGTH X 768]

      # unique_word_ids = list(set(word_ids))
      contracted_out = torch.empty((1, max_length, 768), device=sequence.device)  # match device # alternatively len(unique_word_ids) instead of max_length

      unique_index = 0  # separate index for contracted_out
      seen_word_ids = set()
      for idx, word_id in enumerate(word_ids):
          if word_id not in seen_word_ids:
              seen_word_ids.add(word_id) # add to seen ids to skip at next iteration(s)
              contracted_out[0, unique_index, :] = sequence[0, idx, :]
              unique_index += 1

      return contracted_out

def contract_batch(tokenizer_out, model_out, max_length):
    '''
    tokenizer_out: output of Bert Fast tokenizer, i.e. list containing dictionaries with keys 'input_ids', 'token_type_ids', 'attention_mask'
    model_out: output of Bert Fast model, i.e. list containing each contextualized encoding of a sentence
    '''
    with torch.no_grad():

      word_ids = [example.word_ids() for example in tokenizer_out]  # list containing lists, i.e. mappings from subword_token indices to full word indices in original sentence
      # tensor to be filled with batch of sequences [BATCH_SIZE x SEQ_LEN x 768]
      contracted_batch = torch.empty((len(tokenizer_out), max_length, 768), device=model_out[0].last_hidden_state.device)

      for idx, example in enumerate(model_out):
          tensor = example.last_hidden_state
          contracted_batch[idx] = contract_single_sequence(tensor, word_ids[idx], max_length=max_length)

      return contracted_batch


def get_and_contract_BERT(model, tokenizer, input, include_root=True):
  '''
  model: the BERT model used to compute contextualized embeddings (linked to tokenizer)
  tokenizer: the tokenizer used to extract word ids (linked to model)
  input: a list of lists of sentences containing words
  max_length: the upper bound for lengths of sentences s.t. they will be considered
  '''

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = model.to(device) # move model to gpu if possible

  max_length = len(max(input, key=len)) # length of the lingest sequence in the batch (used for padding)

  if include_root: # if we include root predictions
    tokenizer_out = list( # create a list...
                          map( # of the iterator obtained by applying the following lambda function to 'input' parameter
                              lambda sent: tokenizer(sent + ['[PAD]' for i in range(max_length - len(sent)) if len(sent) < max_length], # note that '<ROOT>' will play the role of the CLS token c.f. BiaffineGraphBasedParser constructor
                                                    add_special_tokens=False,
                                                    is_split_into_words=True,
                                                    return_tensors='pt'
                                                    ).to(device), # need to move tensor because by default py tensors are move to the cpu by tokenizer
                              input)) # in this case '<ROOT>' will be a special token of the tokenizer which has to be learned by the model
  else:
    max_length = max_length - 1 # note that not including root takes one off the maximum length sequences will be padded to
    tokenizer_out = list(# create a list...
                          map(# of the iterator obtained by applying the following lambda function to 'input' parameter
                              lambda sent: tokenizer(sent[1:] + ['[PAD]' for i in range(max_length - len(sent)) if len(sent) < max_length], # sent[1:] removes '<ROOT>' token
                                                     add_special_tokens=False,
                                                     is_split_into_words=True,
                                                     return_tensors='pt'
                                                     ).to(device), # need to move tensor because by default py tensors are move to the cpu by tokenizer
                              input)) # in this case we do not predict root token. Note that this requires setting include_root to false when preparing the data

  model_out = list(map(lambda inputs: model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]), tokenizer_out)) # list of padded sequence representations

  # [BATCH_SIZE x MAX_LENGTH x 768]
  out = contract_batch(tokenizer_out, model_out, max_length=max_length) # contract all sequences only retaining first token in case of subwordtokens cf. other contract functions

  return out


# this is the last layer in the model, outputting an adjacency matrix of PADDED sentences
class Biaffine(nn.Module):
    '''
    biaffine scorer --> used to compute v.U.u + W.(concat(v, u)) + b
    '''

    # input batches number, sequence length, and embeddings_size
    def __init__(self, d):  # d is the size of a word vector

        # initialize superclass of Biaffine, i.e. nn.Module
        super(Biaffine, self).__init__()

        # now we define the parameters. Their size are implicit in the definition of the biaffine scorer
        # --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b

        # a d x d matrix to return a scalar for v.U.u
        self.U = nn.Parameter(torch.empty(d + 1, d))  # [d + 1 x d] + 1 for integrated bias
        init.xavier_normal_(self.U)

        # a single vector of adequate size to return a scalar for W.(concat(v, u))
        self.W = nn.Parameter(torch.empty(2 * d + 1))  # [2*d]
        init.zeros_(self.W)

    def forward(self, D, H):
        '''
        :param H: A tensor containing head representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b
        # -------------------- v.U.u --------------------

        # for integrated bias concatenate ones along the word encoding axis of the heads matrix H
        ones = torch.ones(D.size(0), D.size(1), 1, device=self.U.device)
        D = torch.cat((D, ones), dim=2) # get a one
        U_product = torch.einsum("bxi,ij,byj->bxy", D, self.U, H)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        # Expand P and Q to include the necessary dimensions for concatenation
        # -------------------- W.(concat(v, u)) --------------------
        # we want a tensor of size [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH x d] to access all head-dep concatenations in a given sentence
        D_expanded = D.unsqueeze(2).expand(-1, -1, H.shape[1], -1)  # H is expanded to [BATCH_SIZE, SEQ_LENGTH, 1, d]
        H_expanded = H.unsqueeze(1).expand(-1, D.shape[1], -1, -1)  # D to [BATCH_SIZE, 1, SEQ_LENGTH, d]

        # Concatenate along the last dimension
        # concat_batch[i, j, k,:]  # will correspond to the concatenation of the dependent word j and head word k at the ith sentence in the batch
        concat_batch = torch.cat((D_expanded, H_expanded), dim=3)  # Final shape: [BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH, 2*d]

        # again we can leverage torch tensor operations by using einsum
        # W_product[i, j, k] is defined as W.(concat(v, u)) at sentence i where v is head repr of word j and u is dep repr of word k
        W_product = torch.einsum("ijkl,l->ijk", concat_batch, self.W)  # shape [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        # -------------------- FULL SCORE --------------------
        out = U_product + W_product  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        return out


class SimpleBiaffine(nn.Module):
    '''simplification of the Biaffine scorer --> used to compute v.U.u + b
    '''

    def __init__(self, d):
        # initialize superclass of Biaffine, i.e. nn.Module
        super(SimpleBiaffine, self).__init__()

        # a d x d matrix to return a scalar for v.U.u
        self.U = nn.Parameter(torch.Tensor(d + 1, d))  # [d + 1 x d + 1] (d + 1 integrated bias)
        init.xavier_normal_(self.U)
        # init.zeros_(self.U)


    def forward(self, D, H):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> SimpleBiaffine(v, u) := v.U.u + b

        # integrate bias
        ones = torch.ones(D.size(0), D.size(1), 1, device=self.U.device) # H and D have the same size, hence 'ones' can be reused
        D = torch.cat((D, ones), dim=2) # add bias

        # -------------------- v.U.u --------------------
        U_product = torch.einsum("bxi,ij,byj->bxy", D, self.U, H)  # [BATCH_SIZE x NB_CLASSES x SEQ_LENGTH x SEQ_LENGTH]
        # U_product.squeeze(1) # we only have one output class

        return U_product


class Bilinear(nn.Module):
    ''' Version of SimpleBiaffine scorer with no bias --> computes v.U.u
    '''

    def __init__(self, d):
        # initialize superclass of Bilinear, i.e. nn.Module
        super(Bilinear, self).__init__()

        # now we define the parameters. Their size are implicit in the definition of the bilinear scorer
        # --> Bilinear(v, u) := v.U.u

        # a d x d matrix to return a scalar for v.U.u
        self.U = nn.Parameter(torch.empty(d, d))  # [d x d]
        init.xavier_normal_(self.U)

    def forward(self, D, H):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # v.U.u
        out = torch.einsum("bxi,ij,byj->bxy", D, self.U, H)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        return out

class SplitMLP(nn.Module):
    '''used to split incoming word representations into head and dependent representations'''

    # default behavior should be BiLSTM_layer=output_dim
    def __init__(self, bilstm_hidden_size, hidden_dim, output_dim, dropout=0):
        super(SplitMLP, self).__init__()

        # first weight matrix [BiLSTM_layer + 1 x hidden_dim] (bias is intergated)
        self.w_1 = nn.Linear(bilstm_hidden_size, hidden_dim)

        # second weight matrix [hidden_dim + 1 x output_dim] (bias is integrated)
        self.w_2 = nn.Linear(hidden_dim, output_dim)

        # we use dropout at the last layer during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, biLSTM_layer):
        ''' expected defautl behavior is for args BiLSTM_layer = output_dim
        :param BiLSTM last state [BACTH_SIZE x SEQ_LENGTH x BiLSTM_LAYER_SIZE]
        :return: head/dependent representations [BATCH_SIZE x SEQ_LENGTH x BiLSTM_LAYER_SIZE]
        '''
        # check cuda
        if torch.cuda.is_available():
          device = torch.device("cuda")
        else:
          device = torch.device("cpu")

        # first linear transformation
        out = self.w_1(biLSTM_layer)

        # non-linear activation function
        out = torch.relu(out)

        # second linear transformation
        out = self.w_2(out)

        # apply dropout (note we can apply dropout after all linear transformations because out is passed to the biaffine scorer)
        self.dropout(out)

        return out # [BATCH_SIZE x SEQ_LENGTH x d]

class GraphBasedParser(nn.Module):

    def __init__(self, MLP_hidden_layer=600, d=600, embed="static", vocab=None, POS_Embeddings=False, scorer="SimpleBiaffine"):
        super(GraphBasedParser, self).__init__()

        # when we use GloVe pretrained embeddings
        if embed=="static":
          self.embed = "static"
          print("loading pretrained embeddings...")
          glove = GloVe()
          self.embeddings = nn.Embedding.from_pretrained(glove.weights, freeze=False) # glove.weights returns a tensor of size [|V| x EMBEDDINGS_DIM]
          self.w2i = glove.w2i
          self.i2w = glove.vocab
          print("embeddings loaded!")

        # when we learn embeddings based on the training set requires to pass vocab as kwarg when initializing Parser
        elif embed == "scratch":
          self.embed="scratch"
          self.embeddings = nn.Embedding(len(vocab), 100)
          self.embeddings.weight.requires_grad = True # enables fine tuning of embeddings
          self.vocab = vocab
          self.w2i = get_w2i(self.vocab)

        # when we use BERT encodings only
        elif embed == "contextual":
          self.embed="contextual"
          self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # automatically yields FastTokenizer if possible

          self.bert_tokenizer.add_tokens('<ROOT>', special_tokens=True) # add root token as a special token
          self.bert_model = BertModel.from_pretrained("bert-base-uncased")  # loading respective BERT model
          self.bert_model.resize_token_embeddings(len(self.bert_tokenizer)) # since we added root token
          print("BERT model and tokenizer loaded!")

        # when BERT encodings and pretrained embeddings are concatenated
        elif embed == "concat":
          self.embed = "concat"
          # load pretrained embeddings
          print("loading pretrained embeddings...")
          glove = GloVe()
          self.embeddings = nn.Embedding.from_pretrained(glove.weights, freeze=False) # glove.weights returns a tensor of size [|V| x EMBEDDINGS_DIM]
          self.w2i = glove.w2i
          self.i2w = glove.vocab
          print("embeddings loaded!")

          # load BERT model and tokenizer
          self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # automatically yields FastTokenizer if possible
          self.bert_tokenizer.add_tokens('<ROOT>', special_tokens=True) # add root token as a special token
          self.bert_model = BertModel.from_pretrained("bert-base-uncased")  # loading respective BERT model
          self.bert_model.resize_token_embeddings(len(self.bert_tokenizer)) # since we added root token
          print("BERT model and tokenizer loaded!")

        # case of invalid choice of argument for word encoding
        else:
          raise ValueError("Invalid kwarg for 'embed'. Must choose 'static', 'contextual' or 'scratch'.")

        # initialize biLSTM
        if self.embed == "contextual":
          self.bilstm = nn.LSTM(768, hidden_size=d, num_layers=3, batch_first=True, bidirectional=True, dropout=0.33) # need to manually set size input dim to BERT size
        elif self.embed == "concat":
          self.bilstm = nn.LSTM(868, hidden_size=d, num_layers=3, batch_first=True, bidirectional=True, dropout=0.33) # need to manually set size input dim to BERT size
        else:
          self.bilstm = nn.LSTM(self.embeddings.weight.shape[1], hidden_size=d, num_layers=3, batch_first=True, bidirectional=True, dropout=0.33)

        # initialize two MLPs which yield head and dependent representations
        self.MLP_head = SplitMLP(bilstm_hidden_size=d * 2, hidden_dim=d, output_dim=d, dropout=0.33)
        self.MLP_dep = SplitMLP(bilstm_hidden_size=d * 2, hidden_dim=d, output_dim=d, dropout=0.33)

        # initialize the scorer (Biaffine function)
        if scorer == "SimpleBiaffine":
            self.scorer = SimpleBiaffine(d)
            # self.scorer = nn.Bilinear(d, d, 110)
        elif scorer == "Biaffine":
            self.scorer = Biaffine(d)
        elif scorer == "Bilinear":
            self.scorer = Bilinear(d)
        else:
          raise ValueError("Invalid kwarg. Please choose scorer= 'Biaffine', 'SimpleBiaffine', or 'Bilinear'. Default is 'SimpleBiaffine'")



    def forward(self, X, is_sorted=False, include_root=True): # vectorization/sent_lengths for dm dataset is provided in preprocessing file

        # check cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # step 0: ------------------------------------------VECTORIZING AND PADDING------------------------------------------


        # get seq lengths tensor of size [BATCH_SIZE]
        if include_root:
          X_lengths = get_sentence_lengths(X, include_root=True).to(device)
        else:
          X_lengths = get_sentence_lengths(X, include_root=False).to(device)

        # padding vectorized sequences
        if self.embed == "static" or self.embed == "concat":
          vectorized_X = vectorize(X, self.w2i)
          vectorized_X = pad_vect_sentences(vectorized_X, self.w2i)
          vectorized_X = vectorized_X.to(device)
        elif self.embed == "scratch":
          vectorized_X = vectorize(X, self.w2i)
          vectorized_X = pad_vect_sentences(vectorized_X, get_w2i(self.vocab))
          vectorized_X = vectorized_X.to(device)


        # step 1: ------------------------------------------ENCODING------------------------------------------

         # needs to be placed out of no_grad context because self attention is computed in this function (contains no_grad when necessary inside of function)
        if self.embed == "contextual":
          X = get_and_contract_BERT(self.bert_model, self.bert_tokenizer, X, include_root=include_root) # returns padded sequence representations

        # tensor of size [BACTH_SIZE x SEQ_LENGTH x EMBED_SIZE]
        # no need to sort in decreasing order since the examples were already sorted in the very beginning to allow mapping to Y_train, i.e. adjacency matrices
        if self.embed == "static" or self.embed == "scratch":
          X = self.embeddings(vectorized_X)

        if self.embed == "concat":
          X_1 = get_and_contract_BERT(self.bert_model, self.bert_tokenizer, X, include_root=include_root) # returns padded sequence representations
          X_2 = self.embeddings(vectorized_X)
          X = torch.cat((X_1, X_2), dim=-1) # tensor of shape [BATCH_SIZE x SEQ_LENGTH x 868]

        X = pack_padded_sequence(X, X_lengths.cpu().numpy(), enforce_sorted=is_sorted, batch_first=True)
        X = X.to(device)
        # sizes ...
        X_encoded, (ht, ct) = self.bilstm(X)
        del ht, ct # not needed

        # Assuming `X_encoded` is the PackedSequence object from the BiLSTM
        # Unpack the sequences
        X_encoded, input_sizes_padded = pad_packed_sequence(X_encoded, batch_first=True)

        X_encoded = X_encoded.to(device) # move to GPU if available
        X_lengths = X_lengths.to(device)

        # step 2: ------------------------------------------SPLITTING LAST RECURRENT STATE (HEAD & DEP)------------------------------------------
        # split recurrent states
        heads = self.MLP_head(X_encoded)
        deps = self.MLP_dep(X_encoded)

        # step 3: -------------------------------------------SCORING CANDIDATE ARCS------------------------------------------
        out = self.scorer(deps, heads)

        return out # outputs logits

    def predict(self, X, is_sorted=False, include_root=True):
      '''given an input list of lists of sentences return all their adjacency matrices representing semantic relations'''

      # take sigmoid over logit and then round
      with torch.no_grad():
        pred = torch.round(torch.sigmoid(self.forward(X, is_sorted=is_sorted, include_root=include_root)))
      return pred # outputs rounded probabilities, i.e. 1 or 0

    def serialize(self, file_path):
      '''
      method which saves the current model to file_path
      '''

      try:
        torch.save(self, file_path)
      except Exception as e:
        print(f'An error occurred while saving the model: {e}')
