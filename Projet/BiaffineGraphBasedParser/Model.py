import torch
import torch.nn as nn


# CONVENTIONS
# SEQ_LENGTH is the uniform length of (padded) sentences in a batch
# d is the number of dimension of every word vector after MLP split
# e is the number of dimensions of every word vector before MLP split
# BATCH_SIZE is the number of sentences
# when using einsum we use the same letter in large and small caps when referring to axis of same length, e.g. s and S may refer to SEQ_LENGTH axis

# this is the last layer in the model, outputting an adjacency matrix of PADDED sentences
class Biaffine(nn.Module):
    '''
    biaffine scorer --> computes v.U.u + W.(concat(v, u)) + b
    '''

    # input batches number, sequence length, and embeddings_size
    def __init__(self, d):  # d is the size of a word vector

        # initialize superclass of Biaffine, i.e. nn.Module
        super(Biaffine, self).__init__()

        # now we define the parameters. Their size are implicit in the definition of the biaffine scorer
        # --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b

        # a d x d matrix to return a scalar for v.U.u
        self.U = nn.Parameter(torch.zeros(d, d))  # [d x d]

        # a single vector of adequate size to return a scalar for W.(concat(v, u))
        self.W = nn.Parameter(torch.zeros(2 * d))  # [2*d]

        self.b = nn.Parameter(torch.zeros(1))  # scalar

    def forward(self, H, D):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> Biaffine(v, u) := v.U.u + W.(concat(v, u)) + b

        # -------------------- v.U.u --------------------
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
        out = U_product + W_product + self.b  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        return out


class SimpleBiaffine(nn.Module):
    '''simplification of the Biaffine scorer --> computes v.U.u + b
    '''

    def __init__(self, d):
        # initialize superclass of Biaffine, i.e. nn.Module
        super(SimpleBiaffine, self).__init__()

        # a d x d matrix to return a scalar for v.U.u
        self.U = nn.Parameter(torch.zeros(d, d))  # [d x d]

        self.b = nn.Parameter(torch.zeros(1))  # scalar


    def forward(self, H, D):
        '''
        :param H: A tensor containing heads representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :param D: A tensor containing dependent representation of words size [BATCH_SIZE x SEQ_LENGTH x d]
        :return: A tensor containing the scores of each head dependent pairs for each sentence [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]
        '''

        # Recall --> SimpleBiaffine(v, u) := v.U.u + b

        # -------------------- v.U.u --------------------
        U_product = torch.einsum("bsd, tT, bSD-> bSs", H, self.U, D)  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        # -------------------- FULL SCORE --------------------
        out = U_product + self.b  # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH]

        return out


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
    pass  # TODO

class Model(nn.Module):
    '''Combines building blocks in the Biaffine graph-based parserpipeline from preprocessed inputs to adjacency matrix output'''
    pass # TODO

def fit_model():
    '''
    used to fit the model in an end-to-end manner
    :return: 
    '''
    pass # TODO
