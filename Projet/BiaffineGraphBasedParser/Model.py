import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.distilbert.modeling_distilbert import Embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init


# CONVENTIONS
# SEQ_LENGTH (or SEQUENCE_LENGTH) is the uniform length of (padded) sentences in a batch
# d is the number of dimension of every word vector after MLP split
# e is the number of dimensions of every word vector before MLP split
# BATCH_SIZE is the number of sentences
# when using einsum we use the same letter in large and small caps when referring to axis of same length, e.g. s and S may refer to SEQ_LENGTH axis

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
        self.U = nn.Parameter(torch.zeros(d + 1, d))  # [d + 1 x d] + 1 for integrated bias

        # a single vector of adequate size to return a scalar for W.(concat(v, u))
        self.W = nn.Parameter(torch.zeros(2 * d + 1))  # [2*d]

    def forward(self, H, D):
        '''
        :param H: A tensor containing head representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b
        # -------------------- v.U.u --------------------

        # for integrated bias concatenate ones along the word encoding axis of the heads matrix H
        ones = torch.ones(H.size(0), H.size(1), 1, device=self.U.device)
        H = torch.cat((H, ones), dim=2) # get a one
        U_product = torch.einsum("bsd, tT, bSD-> bSs", H, self.U, D)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        # Expand P and Q to include the necessary dimensions for concatenation
        # -------------------- W.(concat(v, u)) --------------------
        # we want a tensor of size [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH x d] to access all head-dep concatenations in a given sentence
        H_expanded = H.unsqueeze(2).expand(-1, -1, D.shape[1], -1)  # H is expanded to [BATCH_SIZE, SEQ_LENGTH, 1, d]
        D_expanded = D.unsqueeze(1).expand(-1, H.shape[1], -1, -1)  # D to [BATCH_SIZE, 1, SEQ_LENGTH, d]

        # Concatenate along the last dimension
        # concat_batch[i, j, k,:]  # will correspond to the concatenation of the dependent word j and head word k at the ith sentence in the batch
        concat_batch = torch.cat((H_expanded, D_expanded), dim=3)  # Final shape: [BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH, 2*d]

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
        self.U = nn.Parameter(torch.Tensor(d + 1, d))  # [d + 1 x d] (+ 1 along dim 0 for integrated bias)
        init.zeros_(self.U)


    def forward(self, H, D):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> SimpleBiaffine(v, u) := v.U.u + b

        # integrate bias
        ones = torch.ones(H.size(0), H.size(1), 1, device=self.U.device)
        H = torch.cat((H, ones), dim=2) # get a one

        # -------------------- v.U.u --------------------
        U_product = torch.einsum("bsd, tT, bSD-> bSs", H, self.U, D)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]


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
        self.U = nn.Parameter(torch.zeros(d, d))  # [d x d]

    def forward(self, H, D):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # v.U.u
        out = torch.einsum("bsd, tT, bSD-> bSs", H, self.U, D)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

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

        # integrate bias ones [BATCH_SIZE, SEQ_LENGTH, 1]
        # ones = torch.ones(biLSTM_layer.shape[0], biLSTM_layer.shape[1], 1, device=biLSTM_layer.device)
        # biLSTM_layer = torch.cat((biLSTM_layer, ones), -1) # concatenate along the last dimension

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

    def __init__(self, MLP_hidden_layer=600, d=600, embed="pretrained", vocab=None, POS_Embeddings=False, scorer="SimpleBiaffine"):
        super(GraphBasedParser, self).__init__()

        # initialize embeddings
        if embed=="pretrained":
          self.embed = "pretrained"
          print("loading pretrained embeddings...")
          glove = GloVe()
          self.embeddings = nn.Embedding(len(glove.vocab), glove.embed_dim)
          self.embeddings.weight = nn.Parameter(glove.weights)
          self.w2i = glove.w2i
          print("embeddings loaded!")
        else:
          self.embed="scratch"
          self.embeddings = nn.Embedding(len(vocab), 40)
          self.vocab = vocab
          self.w2i = get_w2i(self.vocab)

        # initialize biLSTM
        self.bilstm = nn.LSTM(self.embeddings.weight.shape[1], hidden_size=d, num_layers=3, batch_first=True, bidirectional=True)

        # initialize two MLPs which yield head and dependent representations
        self.MLP_head = SplitMLP(bilstm_hidden_size=d * 2, hidden_dim=d, output_dim=d, dropout=0.25)
        self.MLP_dep = SplitMLP(bilstm_hidden_size=d * 2, hidden_dim=d, output_dim=d, dropout=0.25)

        # initialize the scorer (Biaffine function)
        if scorer == "SimpleBiaffine":
            self.scorer = SimpleBiaffine(d)
        elif scorer == "Biaffine":
            self.scorer = Biaffine(d)
        elif scorer == "Bilinear":
            self.scorer = Bilinear(d)
        else:
          raise ValueError("Incorrect kwarg. Please choose scorer= 'Biaffine', 'SimpleBiaffine', or 'Bilinear'. Default is 'SimpleBiaffine'")



    def forward(self, X): # vectorization/sent_lengths for dm dataset is provided in preprocessing file

        # check cuda
        if torch.cuda.is_available():
          device = torch.device("cuda")
        else:
          device = torch.device("cpu")

        # step 0: ------------------------------------------SORTING, VECTORIZING AND PADDING------------------------------------------

        # retrieving original order on sentences and sorting them deacreasingly by their length
        sorted_lst_with_idx = sorted(enumerate(X), key=lambda x: -len(x[1]))
        sorted_X = [x[1] for x in sorted_lst_with_idx]
        original_ordering = torch.tensor([x[0] for x in sorted_lst_with_idx], device=device)

        # get seq lengths tensor of size [BATCH_SIZE]
        X_lengths = get_sentence_lengths(sorted_X)
        X_lengths = X_lengths.to(device)
        # print(X_lengths)
        # vectorizing sorted sentences
        vectorized_X = vectorize(sorted_X, self.w2i)

        # padding vectorized sequences
        if self.embed == "pretrained":
          vectorized_X = pad_vect_sentences(vectorized_X, self.w2i)
        elif self.embed == "scratch":
          vectorized_X = pad_vect_sentences(vectorized_X, get_w2i(self.vocab))

        vectorized_X = vectorized_X.to(device)



        # step 1: ------------------------------------------ENCODING------------------------------------------

        # tensor of size [BACTH_SIZE x SEQ_LENGTH x EMBED_SIZE]
        # no need to sort in decreasing order since the examples were already sorted in the very beginning to allow mapping to Y_train, i.e. adjacency matrices
        X = self.embeddings(vectorized_X)

        # packed X_train of size [batch_sum_seq_len X EMBEDDINGS_SIZE] --> used for biLSTM encoding only
        X = pack_padded_sequence(X, X_lengths.cpu().numpy(), enforce_sorted=True, batch_first=True)
        X = X.to(device)
        # sizes ...
        X_encoded, (ht, ct) = self.bilstm(X)
        del ht, ct # not needed

        # unpack X_Train packed (This sequence can change the size of the sequence length depending on MAX_LENGTH in sent_lengths)
        X_encoded, input_sizes = pad_packed_sequence(X_encoded, batch_first=True)

        # step 1.5 ------------------------------------------restoring initial order of batch------------------------------------------
        X_encoded = X_encoded[original_ordering].to(device)
        X_lengths = X_lengths[original_ordering].to(device)

        # step 2: ------------------------------------------SPLITTING LAST RECURRENT STATE (HEAD & DEP)------------------------------------------
        # split recurrent states
        heads = self.MLP_head(X_encoded)
        deps = self.MLP_dep(X_encoded)

        # step 3: -------------------------------------------SCORING CANDIDATE ARCS------------------------------------------
        out = self.scorer(heads, deps)

        # we ignore edges involving pad tokens by multipyling them with zero
        # --> done via elementwise multiplication with a mask tensor
        mask = get_mask(out, X_lengths)
        mask = mask.to(out.device)
        out = out * mask

        return out
