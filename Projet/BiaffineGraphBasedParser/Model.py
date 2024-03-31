import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.distilbert.modeling_distilbert import Embeddings


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
        self.W = nn.Parameter(torch.zeros(2 * d))  # [2*d]

    def forward(self, H, D):
        '''
        :param H: A tensor containing head representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representations of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b

        # -------------------- v.U.u --------------------
        U_product = torch.einsum("bsd, tT, bSD-> bSs", H, self.U, D)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        # for integrated bias concatenate ones along the word encoding axis of the heads matrix H
        ones = torch.ones(H.size(0), H.size(1), 1)
        H = H.cat((H, ones), dim=2) # get a one

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
        self.U = nn.Parameter(torch.zeros(d + 1, d))  # [d + 1 x d] (+ 1 along dim 0 for integrated bias)

    def forward(self, H, D):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> SimpleBiaffine(v, u) := v.U.u + b

        # integrate bias
        ones = torch.ones(H.size(0), H.size(1), 1)
        H = H.cat((H, ones), dim=2) # get a one

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
        self.w_1 = nn.Linear(bilstm_hidden_size + 1, hidden_dim)

        # second weight matrix [hidden_dim + 1 x output_dim] (bias is integrated)
        self.w_2 = nn.Linear(hidden_dim + 1, output_dim)

        # we use dropout at the last layer during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, biLSTM_layer):
        ''' expected defautl behavior is for args BiLSTM_layer = output_dim
        :param BiLSTM last state [BACTH_SIZE x SEQ_LENGTH x BiLSTM_LAYER_SIZE]
        :return: head/dependent representations [BATCH_SIZE x SEQ_LENGTH x BiLSTM_LAYER_SIZE]
        '''

        # integrate bias ones [BATCH_SIZE, SEQ_LENGTH, 1]
        ones = torch.ones(self.biLSTM_layer.shape[0], self.biLSTM_layer.shape[1], 1)
        biLSTM_layer = torch.cat((biLSTM_layer, ones), -1) # concatenate along the last dimension

        # first linear transformation
        out = self.w_1(biLSTM_layer)

        # non-linear activation function
        out = torch.relu(out)

        # integrate bias ones
        out = torch.cat((out, ones), -1)

        # second linear transformation
        out = self.w_2(out)

        # apply dropout (note we can apply dropout after all linear transformations because out is passed to the biaffine scorer)
        self.dropout(out)

        return out # [BATCH_SIZE x SEQ_LENGTH x d]

class GraphBasedParser(nn.Module):
    def __init__(self, MLP_hidden_layer, d, vocab, embeddings="pretrained", POS_Embeddings=False, scorer="SimpleBiaffine", train=False):
        super(GraphBasedParser, self).__init__()

        if embeddings == "pretrained":
            self.embeddings = nn.Embedding(torch.load("glove.pt"))
        else:
            self.embeddings = nn.Embedding(len(vocab), 100)

        self.bilstm = nn.LSTM(d, hidden_size=600, num_layers=3, batch_first=True, bidirectional=True)

        if train:
            self.MLP_head = SplitMLP(d, bilstm_hidden_size=self.bilstm.hidden_size * 2, hidden_dim=600, dropout=0.25)
            self.MLP_head = SplitMLP(d, bilstm_hidden_size=self.bilstm.hidden_size * 2, hidden_dim=600, dropout=0.25)
        else:
            self.MLP_head =SplitMLP(d, bilstm_hidden_size=self.bilstm.hidden_size * 2, hidden_dim=600)
            self.MLP_dep = SplitMLP(d, bilstm_hidden_size=self.bilstm.hidden_size * 2, hidden_dim=600)
            
        if scorer == "SimpleBiaffine":
            self.scorer = SimpleBiaffine(d)
        elif scorer == "Biaffine":
            self.scorer = Biaffine(d)
        else:
            self.scorer = Bilinear(d)
            

    def forward(self):
        pass # TODO
    

def fit_model(X_train):
    '''used to fit the model in an end-to-end manner'''
    pass # TODO
