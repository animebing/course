import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    normal_scores = exp_scores / np.sum(exp_scores)

    loss += -np.log(normal_scores[y[i]])
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += (normal_scores[j] - 1)*X[i].T
      else:
        dW[:, j] += normal_scores[j]*X[i].T

  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  dW /= num_train
  dW += reg*W


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  max_scores = np.max(scores, axis=1)
  max_scores = max_scores[:, np.newaxis]
  scores -= max_scores

  exp_scores = np.exp(scores)

  sum_scores = np.sum(exp_scores, axis=1)
  sum_scores.shape = (num_train, 1)

  normal_scores = exp_scores / sum_scores

  loss = -np.sum(np.log(normal_scores[range(num_train), y]))
  loss /= num_train
  #print 'data loss ', loss
  loss += 0.5*reg*np.sum(W*W)
  #print 'regularization loss ', 0.5*reg*np.sum(W*W)

  normal_scores[range(num_train), y] -= 1
  
  """
  for i in xrange(num_train):
    dW += np.dot(X[i:(i+1)].T, normal_scores[i:(i+1)])
  """
  dW = np.dot(X.T, normal_scores)
  dW /= num_train
  dW += reg*W 



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

