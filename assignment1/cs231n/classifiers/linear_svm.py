import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]  # correct class score
    for j in range(num_classes):
      if j == y[i]:  # except correct class
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # if yi=2 is correct
        # L_i = max(0, x_i1.w_11 + x_i2.w_12 + x_i3.w_13 *** +x_iD.w_1D - x_i1.w_yi1 - x_i2.w_yi2 - x_i3.w_yi3 *** - x_iD.w_yiD)
        #     + 0
        #     + max(0, x_i1.w_31 + x_i2.w_32 + x_i3.w_33 *** +x_iD.w_3D - x_i1.w_yi1 - x_i2.w_yi2 - x_i3.w_yi3 *** - x_iD.w_yiD)
        dW[:, y[i]] -= X[i, :]
        dW[:, j] += X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
#   loss += reg * np.sum(W * W)
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape, dtype=np.float64) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W).astype(np.float64)
  yi_scores = scores[np.arange(scores.shape[0]), y]  # http://stackoverflow.com/a/23435843/459241 
  margins = np.maximum(0, scores - yi_scores.reshape(-1, 1) + 1)  # all score score - correct class score
  margins[np.arange(num_train), y] = 0  # necessary! correct calss should not to be calculated => 0
  
  loss = np.mean(margins)
  if loss is np.nan:
    print('loss is nan')
    print('margins:\n', margins)
  penalty = 0.5 * reg * np.sum(W**2)
  if penalty is np.nan:
    print('penalty is nan')
  loss += penalty
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  # Average
  dW /= num_train

  # Regularize
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
