import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------FUNCTION TO RETRIEVE MASK-------------------------------
def get_mask(examples_tensor, sent_lengths):
  '''given a tensor of size BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH return a mask tensor containing
  '''

  mask = torch.zeros(examples_tensor.shape, dtype=torch.int32)

  for idx, sent_len in enumerate(sent_lengths):
    mask[idx, :sent_len, :sent_len] = torch.ones(sent_len, sent_len) # add ones up until the length of the actual sentence

  return mask

# -------------------------------customized loss function (inherits: torch.nn.BCEWithLogitsLoss-------------------------------
class BCEWithLogitsLoss_masked(nn.BCEWithLogitsLoss):

  def __init__(self, weight=None, size_average=None, reduce=False, reduction='none', pos_weight=None):
    super(BCEWithLogitsLoss_masked, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)

  def forward(self, pred, target, mask):
    # note that pred is already expected to be masked (cf. 'out' of forward in GraphBasedParser model)

    # calculate the loss wrt each prediction
    loss = mask * (super(BCEWithLogitsLoss_masked, self).forward(pred, target))

    # vector of size [BATCH_SIZE] (mask.sum(dim=(1, 2)) gives us the nb of candidate arcs for each example)
    loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)) # the average loss for each (masked) example

    # scalar
    loss = loss.mean() # average loss over all (masked) predictions

    return loss
    
# ----------------------------------TWO FUNCTIONS USED TO EVALUATE CLASSIFIER (PRED, RECALL, FSCORE, OVERALL ACCURACY)----------------------------------
def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning 0 if the denominator is zero."""
    return numerator / denominator if denominator else 0

def mtrx_accuracy(pred, target, seq_lengths):
  '''given a predicted adjacency matrix, a target adjacency matrix and a seq_length tensor (to mask relevant indices)
  output the percentage of correctly predicted edges out of the relevant edges
  '''

  # get model (if model is on GPU then so is pred)
  device = pred.device
  target = target.to(device)
  seq_lengths = seq_lengths.to(device)

  # first get the mask
  mask = get_mask(target, seq_lengths).to(device)

  #find element-wise matches
  all_matches = (mask * ((target == pred).int())).sum().item()

  # nb of all candidate arcs
  all_candidates = (seq_lengths * seq_lengths).sum().item()

  # total acc
  tot_acc = all_matches / all_candidates

  # get number of all matching 1s (for real candidate arcs)
  num_matching_1s = (mask * (pred * target)).sum().item()

  # number of positives in target is just sum of all 1s in the target tensor)
  num_target_arcs = (target).sum().item()

  # accuracy is given by ration of predicated edges over nb of possible, i.e. relevant edges´
  recall = safe_divide(num_matching_1s, num_target_arcs)

  # total number of 1s that were predicted (in the relevant part of the pred tensor)
  predicted_1s = (pred * mask).sum().item()

  # ratio of true positives over positives prediction
  precision = safe_divide(num_matching_1s, predicted_1s)

  # harmonic mean of precision and recall
  fscore = safe_divide((2*precision * recall), (precision + recall))

  return recall, precision, fscore, tot_acc

# ---------------------------------- TRAINING FUNCTION ----------------------------------
def train_model(
    model, # model the parameters of which are to be optimized

    # sentece lengths not needed currently

    X_train, # training examples to be handed to the classifier
    Y_train, # target values for training set
    X_dev, # dev examples to compute leanring progress (relevant for patience)
    Y_dev, # target values for dev set
    learning_rate=0.01, # learning rate (hyperparam)
    betas=(0, 0.95), # Adam specific hyperparam
    patience=10, # patience (hyperparam)
    max_epochs=10, # maximum number of epochs (hyperparam)
    batch_size=300, # batch_size (hyperparam)
    pos_weight_factor=0.5, # factor multiplied with pos_weight, i.e. weight attatched to positive classes ((num_negatives / num_positives)*pos_weight_factor)
    scheduled=True
    ):

  '''implementation of a training algorithm to update parameters of a model'''

  # first we check if GPU is available
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using the GPU...")
  else:
    device = torch.device("cpu")
    print("GPU is not available. Using the CPU...")

  # move to GPU if possible
  model.to(device)

  Y_dev = Y_dev.to(device)
  Y_train = Y_train.to(device)
  train_sentence_lengths = get_sentence_lengths(X_train).to(device)
  mask = get_mask(Y_train, train_sentence_lengths).to(device) # will be used to only consider loss for real candidate arcs
  dev_sent_lengths = get_sentence_lengths(X_dev).to(device)

  if torch.cuda.is_available():
    print("moved model to GPU")

  total_batches = math.ceil(len(X_train) / batch_size)

  # declare some variables
  breaking_condition_met = False # variable which turns to true iff some breaking condition (early stop, max_epochs etc. is met)
  epoch_counter = 0 # variable to keep track of total number of epochs
  fscore_dev = 0 # variable to keep track of progress for accuracy on dev set (for early stopping)
  initial_patience = patience # variable that is not meant to change as reference in early stopping condition

  # declare optimization algo and loss
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=0.000000003)
  # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  # apply scheduler if desired
  if scheduled:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)

  num_positives = Y_train.sum().item()
  train_sent_lengths = get_sentence_lengths(X_train)
  num_negatives = (train_sent_lengths * train_sent_lengths).sum().item() - num_positives
  # a constant defined via (nb 0s / nb 1s) in the training set. Will favor the underrepresented class when computing the loss
  pos_weights = torch.tensor((num_negatives/num_positives) * pos_weight_factor, device=device) # will be broadcasted across 'target axis' i.e. any axis of target adjacency matrix
  print("pos_weights:", pos_weights)

  if pos_weight_factor:
    criterion = BCEWithLogitsLoss_masked(pos_weight=pos_weights)
  else:
    criterion = BCEWithLogitsLoss_masked()

  while not breaking_condition_met: # turn until some breaking condition is met (patience ran out or max_num_epochs is met)
    print("------------------------------------------")
    print("epoch:", epoch_counter)
    print("---------")

    # shuffle the examples (we need to keep track of indices to order Y_train accordingly)
    indices = list(range(len(X_train)))
    shuffle(indices)
    X_train = [X_train[i] for i in indices]

    # apply the same to target matrices
    indices_tensor = torch.tensor(indices)
    Y_train = Y_train[indices_tensor]
    mask[indices_tensor]

    # getting predictions
    dev_pred= torch.round(torch.sigmoid(model(X_dev)))

    #  # calculate the accuracy on dev set (subset)
    recall, precision, fscore, tot_acc = mtrx_accuracy(dev_pred, Y_dev, dev_sent_lengths)
    print("recall on dev set:", round(recall, 2) * 100, "%")
    print("precision on dev set:", round(precision, 2) * 100, "%")
    print("fscore on dev set:", round(fscore, 4), "/ 1.0")
    print("total match ratio including 1s & 0s:", round(tot_acc, 2) * 100, "%")
    print("---------")
    del dev_pred

    # calculate the accuracy on (subset of) train set
    """
    train_pred = torch.round(torch.sigmoid(model(X_train[:100]))).to(device)
    Y_train_trunc = Y_train[:, :train_pred.size(1), :train_pred.size(2)]
    recall, precision, fscore = mtrx_accuracy(train_pred, Y_train_trunc[:100], get_sentence_lengths(X_train[:100]).to(device))
    print("recall on train set:", round(recall, 2) * 100, "%")
    print("precision on train set:", round(precision, 2) * 100, "%")
    print("fscore on train set:", round(fscore, 4), "/ 1.0")
    print("---------")
    del train_pred
    """

    # check if patience needs to be decreased (when accuracy on dev set decreased)
    if fscore_dev > fscore:
      patience -= 1
    else:
      patience = initial_patience
    print("Patience left:", patience)
    print("------------------------------------------")
    print()

    # update to the current accuracy
    fscore_dev = fscore

    # see if a breaking condition applies
    if not patience or epoch_counter >= max_epochs:
      breaking_condition_met = True

    epoch_counter += 1
    batch_counter = 0

    # iterate over batches
    for i in range(0, len(X_train), batch_size):

      # print statement at each initial
      batch_counter += 1
      if batch_counter % 20 == 0:
        print(f"Batch {batch_counter}/{total_batches} at epoch ({epoch_counter}/max {max_epochs})")


      # reset gradient for each batch
      model.zero_grad()

      # a batch of size BATCH_SIZE x SEQ_LENGTH (batch of vectorized sentences)
      X_batch = X_train[i:i+batch_size]

      # target adjacency matrix batch of shape BATCH_SIZE x SEQ_LENGTH y SEQ_LENGTH
      Y_batch = Y_train[i:i+batch_size]

      # respective sentence lengths
      mask_batch = mask[i:i+batch_size]

      # the shape of seq length can be changed here (in the biLSTM pass)
      # (we can sefely truncate Y_train accordingly in this case)
      logits = model(X_batch)
      Y_batch = Y_batch[:, :logits.size(1), :logits.size(2)] # will truncate the gold matrices to maximum seq_length in current batch
      mask_batch = mask_batch[:, :logits.size(1), :logits.size(2)]

      # compute the (masked) loss
      # loss = criterion(logits, Y_batch, mask_batch)
      loss = criterion(logits, Y_batch, mask_batch)

      # perform backward propagation to get the gradient wrt parameters
      loss.backward()

      # update the parameters according to leanring rate and calculated gradient
      optimizer.step()

    if scheduled:
      scheduler.step() # step after each epoch
