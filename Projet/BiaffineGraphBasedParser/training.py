import torch
import torch.nn as nn
import torch.optim as optim

def get_mask(examples_tensor, sent_lengths):
  '''given a tensor of size BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH return a mask tensor containing
  '''

  mask = torch.zeros(examples_tensor.shape, dtype=torch.int8)

  for idx, sent_len in enumerate(sent_lengths):
    mask[idx, :sent_len, :sent_len] = torch.ones(sent_len, sent_len) # add ones up until the length of the actual sentence

  return mask
    

def mtrx_accuracy(pred, target, seq_lengths):
  '''given a predicted adjacency matrix, a target adjacency matrix and a seq_length tensor (to mask relevant indices)
  output the percentage of correctly predicted edges out of the relevant edges
  '''

  # first get the mask
  mask = get_mask(target, seq_lengths)

  #find element-wise matches
  matches = (target == pred).int()

  # Calculate matching 1s (in practice this mask is not necessary)
  matching_1s = ((matches * target) * mask).sum()

  # number of possible edges is sum of the squares of respective sequence lengths
  possible_matches = (seq_lengths * seq_lengths).sum()

  # invert all values for target and pred, mask relevant tokens and take the sum to figure out how many 0s matched
  matching_0s = (((1 - target) * (1 - pred)) * mask).sum()

  # accuracy is given by ration of predicated edges over nb of possible, i.e. relevant edges´
  accuracy = (matching_1s + matching_0s) / possible_matches

  return accuracy.item()


def train_model(
    model, # model the parameters of which are to be optimized
    sent_lengths, # sent_lengths to be passed to the model
    X_train, # training examples to be handed to the classifier
    Y_train, # target values for training set
    X_dev, # dev examples to compute leanring progress (relevant for patience)
    Y_dev, # target values for dev set
    learning_rate=0.001, # learning rate (hyperparam)
    betas=(0, 0.95), # Adam specific hyperparam
    patience=3, # patience (hyperparam)
    max_epochs=300, # maximum number of epochs (hyperparam)
    batch_size=1000 # batch_size (hyperparam)
    ):

  '''implementation of a training algorithm to update parameters of a model'''

  # declare some variables
  breaking_condition_met = False # variable which turns to true iff some breaking condition (early stop, max_epochs etc. is met)
  epoch_counter = 0 # variable to keep track of total number of epochs
  accuracy_dev = 0 # variable to keep track of progress for accuracy on dev set (for early stopping)
  initial_patience = patience # variable that is not meant to change as reference in early stopping condition

  # declare optimization algo and loss
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
  criterion = nn.BCEWithLogitsLoss()

  while not breaking_condition_met: # turn until some breaking condition is met (patience ran out or max_num_epochs is met)

    print("epoch:", epoch_counter)
    print("Patience left:", patience)

    # create a random permutation of examples (along batch axis)
    perm = torch.randperm(X_train.size(0))
    # then shuffle the examples (and their targets in using same permutation) (and the respective sentence lengths)
    shuff_X_train, shuff_Y_train, shuff_sent_lengths = X_train[perm], Y_train[perm], sent_lengths[perm]

    # get logits
    dev_logits = model(X_dev)
    # get probability of each edge
    dev_probs = torch.sigmoid(dev_logits)
    # derive predictions from probabilities (values >0.5 get predicted as existing edge class 1 otherwise 0)
    dev_pred = torch.round(dev_probs)
    # calculate the accuracy
    accuracy = mtrx_accuracy(dev_pred, Y_dev, sent_lengths)
    print("accuracy on dev set:", accuracy * 100)

    # check if patience needs to be decreased (when accuracy on dev set decreased)
    if accuracy_dev >= accuracy:
      patience -= 1
    else:
      patience = initial_patience

    # update to the current accuracy
    accuracy_dev = accuracy

    # see if a breaking condition applies
    if not patience or epoch_counter >= max_epochs:
      breaking_condition_met = True

    epoch_counter += 1

    # iterate over batches
    for i in range(0, shuff_X_train.shape.size(0), batch_size):

      # reset gradient for each batch
      model.zero_grad()

      # a batch of size BATCH_SIZE x SEQ_LENGTH (batch of vectorized sentences)
      X_batch = shuff_X_train[i:i+batch_size, :]

      # target adjacency matrix batch of shape BATCH_SIZE x SEQ_LENGTH y SEQ_LENGTH
      Y_batch = shuff_Y_train[i:i+batch_size, :, :]

      logits = model(X_batch, sent_lengths)

      loss = criterion(logits, Y_batch)

      # perform backward propagation to get the gradient wrt parameters
      loss.backward()

      # update the parameters according to leanring rate and calculated gradient
      optimizer.step()
