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
  #print "naive: "
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)    # bias here is regarded as a weight
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    margin = scores - correct_class_score + 1
    margin[y[i]] = 0
    margin_bool = margin >0
    margin_bool = margin_bool.astype(float)
    #if i == 1:
    #	print 'naive ', margin_bool
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += -np.sum(margin_bool)*X[i].T
      else:
        dW[:, j] += margin_bool[j]*X[i].T
  dW /= num_train
  dW += reg*W
  

  return loss, dW
 

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implemen
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = scores[range(num_train), y] # shape is (num_train,)
  correct_class_score.shape = (num_train, 1)
  margin = scores - correct_class_score + 1.0
  margin[range(num_train), y] = 0.0;
  margin_bool = margin > 0
  margin_bool = margin_bool.astype(float)

  loss = np.sum(margin*margin_bool)/num_train + 0.5*reg*np.sum(W*W)  # max(0, x), not max()
  

  margin_bool[range(num_train), y] = -np.sum(margin_bool, axis=1)
  """
  for i in xrange(num_train):
  	dW += np.dot(X[i:(i+1)].T, margin_bool[i:(i+1)])
  """

  dW = np.dot(X.T, margin_bool)
  dW /= num_train
  dW += reg*W


  """
  margin_bool[range(num_train), y] = -np.sum(margin_bool, axis=1)
  #print 'vectorized ', margin_bool[1]
  margin_bool.shape = (num_train, 1, num_classes)
  margin_bool_new = np.tile(margin_bool, (1, dim, 1))
  
  #print 'bing', X[0, :5]
  X_new = np.reshape(X, (num_train, dim, 1))   # when reshape, the read and write order is strange
  #print 'haha ', X_new[0, :5, 0]
  X_new = np.tile(X_new, (1, 1, num_classes))
  dW = np.sum(margin_bool_new*X_new, axis=0)/num_train
  dW += reg*W 
	"""


  pass
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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
