import torch
import torch.nn as nn
import torch.optim as optim
import math
from statistics import mean
from random import shuffle

# -------------------------------FUNCTION TO RETRIEVE MASK-------------------------------
def get_mask(examples_tensor, sent_lengths):
  '''
  given a tensor of size BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH return a mask tensor containing ones for candidate edges
  '''

  mask = torch.zeros(examples_tensor.shape, dtype=torch.long)

  for idx, sent_len in enumerate(sent_lengths):
    mask[idx, :sent_len, :sent_len] = torch.ones(sent_len, sent_len) # add ones up until the length of the actual sentence (including root)

  return mask

# -------------------------------customized loss function (inherits: torch.nn.BCEWithLogitsLoss-------------------------------
class BCEWithLogitsLoss_masked(nn.BCEWithLogitsLoss):
  '''
  same as BCEWithLogitsLoss but reuiring a mask in the forward method to ignore irrelevant predictions which ought not contribute to the loss
  kwargs sumloss, avg and weight_1s are added to constructor
  weight_1s: weight multiplied with losses of predictions that are 1 in gold target
  sumloss: if set to True loss will be summed
  avg: if set to True the mean of all losses will be taken
  '''

  def __init__(self, weight=None, size_average=None, reduce=False, reduction='none', pos_weight=None, avg=False, sumloss=True, weight_1s=1):
    super(BCEWithLogitsLoss_masked, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)
    self.avg = avg
    self.sumloss = sumloss
    self.weight_1s = weight_1s

  def forward(self, pred, target, mask):
    '''
    pred: a predictions matrix with logits as values
    target: a target matrix consisting with 1 or 0 as values
    mask: a mask matrix of shape target.shape with 1 as values for predictions which contribute to the loss and otherwise 0
    '''

    device = pred.device
    ones = torch.ones(target.shape).to(device)

    weight = (((target * ones * self.weight_1s) + 1) - target) # returns a tensor with 1s for 0 entries and weight_1s for 1s entries in the target

    # calculate the loss wrt each prediction and apply mask (cancel out non-candidate arcs loss)
    loss = mask * (super(BCEWithLogitsLoss_masked, self).forward(pred, target)) # [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH] (since reduction='none')
    loss = loss * weight

    # vector of size [BATCH_SIZE] (mask.sum(dim=(1, 2)) gives us the nb of candidate arcs for each example)
    # vector of size [BATCH_SIZE] (loss.sum(dim=(1, 2)) gives us the sum of the losses for each example
    if self.avg:
      loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)) # vector of size [BATCH_SIZE] containing the average loss for each example
      # scalar
      loss = loss.mean() # average loss over all (masked) predictions
    elif self.sumloss:
      loss = loss.sum()

    return loss
    
# ----------------------------------FUNCTIONS USED TO EVALUATE CLASSIFIER (PRED, RECALL, FSCORE, OVERALL ACCURACY)----------------------------------
def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning 0 if the denominator is zero."""
    return numerator / denominator if denominator else 0


def mtrx_accuracy(pred, target, seq_lengths):
  '''given a predicted adjacency matrix, a target adjacency matrix and a seq_length tensor (to mask relevant indices)
  output the fraction of correctly predicted edges out of the relevant edges
  '''
  with torch.no_grad():
    # get model (if model is on GPU then so is pred)
    device = pred.device
    target = target.to(device)
    seq_lengths = seq_lengths.to(device)

    # first get the mask
    mask = get_mask(target, seq_lengths).to(device)

    #find element-wise matches
    all_matches = (mask * ((target == pred).int())).sum().item() # returns number of total matches for candidate arcs

    # nb of all candidate arcs
    all_candidates = mask.sum().item() # total number of candidate arcs

    # total acc
    accuracy = all_matches / all_candidates

  return accuracy

def mtrx_recall(pred, target, seq_lengths):
  '''
  given predictions, a tagret tensor and sequence lengths tensor compute recall w.r.t. predictions of real candidate edges (1s)
  --> true positives / (true positives + false negatives)
  '''
  with torch.no_grad():
    # get model (if model is on GPU then so is pred)
    device = pred.device
    target = target.to(device)
    seq_lengths = seq_lengths.to(device)

    # first get the mask
    mask = get_mask(target, seq_lengths).to(device)

    # get number of all matching 1s (for real candidate arcs)
    num_matching_1s = (mask * (pred * target)).sum().item()

    # number of positives in target is just sum of all 1s in the target tensor)
    num_target_arcs = (target).sum().item()

    # accuracy is given by ration of predicated edges over nb of possible, i.e. relevant edges´
    recall = safe_divide(num_matching_1s, num_target_arcs)

  return recall

def mtrx_precision(pred, target, seq_lengths):
  '''
  given predictions, a tagret tensor and sequence lengths tensor compute  precision w.r.t. predictions of real candidate edges (1s)
  --> true positives / (true positives + false positives)
  '''
  with torch.no_grad():
    # get model (if model is on GPU then so is pred)
    device = pred.device
    target = target.to(device)
    seq_lengths = seq_lengths.to(device)

    # first get the mask
    mask = get_mask(target, seq_lengths).to(device)

    # get number of all matching 1s (for real candidate arcs)
    num_matching_1s = (mask * (pred * target)).sum().item()

    # total number of 1s that were predicted (in the relevant part of the pred tensor)
    predicted_1s = (pred * mask).sum().item()

    # ratio of true positives over positives prediction
    precision = safe_divide(num_matching_1s, predicted_1s)

  return precision

def mtrx_fscore(pred, target, seq_lengths):
  '''
  given predictions, a tagret tensor and sequence lengths tensor compute  fscore w.r.t. predictions of real candidate edges (1s)
  --> (2*precision*recall)/(precision + recall)
  '''

  with torch.no_grad():
    # get precision and recall
    precision = mtrx_precision(pred, target, seq_lengths)
    recall = mtrx_recall(pred, target, seq_lengths)

    # harmonic mean of precision and recall
    fscore = safe_divide((2*precision * recall), (precision + recall))

  return fscore

def evaluate_model(model, X_test, Y_test, test_lengths, batch_size='default'):
  '''
  given a model a test set and sequence lengths of examples prints the overall accuracy and precision/recall/fscore w.r.t. 1s predictions
  model: GraphBasedParser object
  X_test: list of lists of sentences split into words preceded by '<ROOT>'
  Y_test: tensor of shape [BATCH_SIZE x SEQ_LEN x SEQ_LEN] containing target edges on semantic dependency graph 
  --> (batch of padded adjacency matrices)
  test_lengths: tensor of shape [BATCH_SIZE] containing sequence lengths
  batch_size : kwarg to be used to indicate batch size in case test sets are too large for inference (applies to models with Biaffine scorer)
  '''
    accs, precs, recs, fscs = [], [], [], []
    if batch_size == 'default':
      batch_size = len(X_test)

    for i in range(0, len(X_test), batch_size):
        pred = model.predict(X_test[i:i+batch_size])
        max_len = test_lengths[i:i+batch_size].max()

        accs.append(mtrx_accuracy(pred, Y_test[i:i+batch_size, :max_len, :max_len], test_lengths[i:i+batch_size]))
        precs.append(mtrx_precision(pred, Y_test[i:i+batch_size, :max_len, :max_len], test_lengths[i:i+batch_size]))
        recs.append(mtrx_recall(pred, Y_test[i:i+batch_size, :max_len, :max_len], test_lengths[i:i+batch_size]))
        fscs.append(mtrx_fscore(pred, Y_test[i:i+batch_size, :max_len, :max_len], test_lengths[i:i+batch_size]))

    mean_acc = mean(accs) * 100
    mean_prec = mean(precs) * 100
    mean_rec = mean(recs) * 100
    mean_fsc = mean(fscs) * 100

    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Mean Precision: {mean_prec:.2f}%")
    print(f"Mean Recall: {mean_rec:.2f}%")
    print(f"Mean F-Score: {mean_fsc:.2f}%")

# ---------------------------------- TRAINING FUNCTION ----------------------------------
def train_model(
    model, # model the parameters of which are to be optimized

    # sentece lengths not needed currently

    X_train, # training examples to be handed to the classifier
    Y_train, # target values for training set
    X_dev, # dev examples to compute leanring progress (relevant for patience)
    Y_dev, # target values for dev set
    learning_rate=1e-4, # learning rate (hyperparam)
    betas=(0.9, 0.9), # Adam specific hyperparam
    patience=10, # patience (hyperparam)
    max_epochs=10, # maximum number of epochs (hyperparam)
    batch_size=300, # batch_size (hyperparam)
    scheduled=False,
    weight_1s=1, # weight attached to loss w.r.t. predictions concerning existing arcs
    loss_lst = None, # when a list is passed it will be filled with the computed losses
    sort_batches=False, # when set to True, batches are sorted in decreasing order of their length each time. Default is False
    learn_root=True
    ):

  '''implementation of a training algorithm to update parameters of a model'''

  steps = 0 # keep count of param updates

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
  train_lengths = get_sentence_lengths(X_train, include_root=learn_root).to(device)
  mask = get_mask(Y_train, train_lengths).to(device) # will be used to only consider loss for real candidate arcs
  dev_lengths = get_sentence_lengths(X_dev, include_root=learn_root).to(device)

  if torch.cuda.is_available():
    print("moved model to GPU")

  total_batches = math.ceil(len(X_train) / batch_size)

  # declare some variables
  breaking_condition_met = False # variable which turns to true iff some breaking condition (early stop, max_epochs etc. is met)
  epoch_counter = 0 # variable to keep track of total number of epochs
  fscore_last = 0 # variable to keep track of progress for accuracy on dev set (for early stopping)
  initial_patience = patience # variable that is not meant to change as reference in early stopping condition
  loss = "-" # not of correct type yet. Will be float during training
  norm_loss = "-" # not of correct type yet. Will be a float during training

  # declare optimization algo and loss
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=0.000000003)
  # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  # apply scheduler if desired
  if scheduled:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)

  criterion = BCEWithLogitsLoss_masked(weight_1s=weight_1s)
  # criterion_dev = BCEWithLogitsLoss_masked()


  while not breaking_condition_met: # turn until some breaking condition is met (patience ran out or max_num_epochs is met)
    print("------------------------------------------")
    print("epoch:", epoch_counter)
    print("---------")

    # shuffle the examples (we need to keep track of indices to order Y_train accordingly)
    with torch.no_grad():

      indices = list(range(len(X_train)))
      shuffle(indices)
      X_train = [X_train[i] for i in indices]

      # apply the same to target matrices
      indices_tensor = torch.tensor(indices)
      Y_train = Y_train[indices_tensor]
      mask[indices_tensor]

      # getting predictions on dev set
      dev_pred = model.predict(X_dev)

      # calculate the accuracy on dev set (subset)
      accuracy = mtrx_accuracy(dev_pred, Y_dev, dev_lengths)
      recall = mtrx_recall(dev_pred, Y_dev, dev_lengths)
      precision = mtrx_precision(dev_pred, Y_dev, dev_lengths)
      fscore = mtrx_fscore(dev_pred, Y_dev, dev_lengths)

      print(f"Total accuracy on dev set: {(accuracy * 100):.2f}%")
      print(f"Recall on dev set: {(recall * 100):.2f}%")
      print(f"Precision on dev set: {(precision * 100):.2f}%")
      print(f"Fscore on dev set: {(fscore * 100):.2f}%")
      print(f"loss on last epoch: {loss}")
      print(f"normalized loss last epoch {norm_loss}")


      print("---------")
      del dev_pred # , train_pred

      # check if patience needs to be decreased (when accuracy on dev set decreased)
      if fscore_last >= fscore:
        patience -= 1
      else:
        patience = initial_patience
        # update to the current accuracy
        fscore_last = fscore
      print("Patience left:", patience)
      print("------------------------------------------")
      print()

      # see if a breaking condition applies
      if not patience or epoch_counter >= max_epochs:
        breaking_condition_met = True

      epoch_counter += 1
      batch_counter = 0

    # in this context gradients are computed
    # iterate over batches
    for i in range(0, len(X_train), batch_size):

      # print statement at each initial
      batch_counter += 1
      if batch_counter % 20 == 0:
        print(f"Batch {batch_counter}/{total_batches} at epoch ({epoch_counter}/max {max_epochs})")


      # reset gradient for each batch
      model.zero_grad()

      with torch.no_grad():

        # a batch of sentences split into words list of lists of words preceded by root token
        # [['<ROOT>', 'this', 'is', 'a', 'sentence'], ['<ROOT>',...], ...]
        X_batch = X_train[i:i+batch_size]

        if sort_batches:
          # sorted(list(enumerate(X_batch)), key=lambda x: -len(x[1])) gives a list containing example_idx, sentence pairs ordered decreasingly by length
          sorted_idx_sentence_lst = sorted(list(enumerate(X_batch)), key=lambda x: -len(x[1]))

          # X_batch_idx the only contains the original idx of a sentence at the position of the sentence in the ordered list
          X_batch_idx = [idx for idx, sentence in sorted_idx_sentence_lst]
          # X_batch is now sorted decreasingly by length
          X_batch = [sentence for idx, sentence in sorted_idx_sentence_lst]

          # ordering needed for target and mask tensor
          ordering = torch.tensor(X_batch_idx).to(device)

        # target adjacency matrix batch of shape BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH
        Y_batch = Y_train[i:i+batch_size]
        if sort_batches:
          Y_batch = Y_batch[ordering] # sort targets as X_batch has been sorted

        # respective sentence lengths
        # a tensor of shape [BATCH_SIZE x SEQ_LENGTH x SEQ_LENGTH] containing 1s for all relevant predictions and 0s for all irrelevant ones
        mask_batch = mask[i:i+batch_size]
        if sort_batches:
          mask_batch = mask[ordering]

      # the shape of seq length can be changed here (in the biLSTM pass)
      # (we can safely truncate Y_train accordingly in this case)
      logits = model(X_batch, is_sorted=sort_batches, include_root=learn_root) # deafault settings for forward
      with torch.no_grad():
        Y_batch = Y_batch[:, :logits.size(1), :logits.size(2)] # will truncate the gold matrices to maximum seq_length in current batch
        mask_batch = mask_batch[:, :logits.size(1), :logits.size(2)]

      # compute the (masked) loss
      loss = criterion(logits, Y_batch, mask_batch)
      # loss_dev = criterion_dev(logits, Y_batch, mask_batch)
      with torch.no_grad():
        norm_loss = loss / mask_batch.sum() # loss relative to the number of candidate arcs in a batch

        if loss_lst is not None:
          loss_lst.append(norm_loss)

      # perform backward propagation to get the gradient wrt parameters
      loss.backward()

      # update the parameters according to leanring rate and calculated gradient
      optimizer.step()

      steps += 1

    if scheduled:
      scheduler.step() # step after each epoch
