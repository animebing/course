import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
  
  """
  [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  """
  

  def __init__(self, num_filters, hidden_dims, filter_size=3, input_dims=[3, 32, 32], num_classes=10,
               reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new MultiLayerConvNet
    CAUTION: the size of input and output feature of conv is identical and stride is 1
    so the pad is necessary and the number of pad equals (filter_size-1)/2
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - num_filters: a list indicating the number of filter of each conv layer
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    num_convs = len(num_filters)
    self.num_convs = num_convs
    self.filter_size = filter_size
    self.reg = reg
    self.dtype = dtype
    self.params = {}
    num_layers = num_convs+len(hidden_dims)+1
    self.num_layers = num_layers
    C, H, W = input_dims
    
    input_c = input_dims[0]
    for i in xrange(num_convs):
      self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i], input_c, filter_size, filter_size)
      self.params['b'+str(i+1)] = np.zeros((num_filters[i]))
      input_c = num_filters[i]

    H_out = H/(2**(num_convs-1))
    W_out = W/(2**(num_convs-1))
    input_before = input_c*H_out*W_out
    for i in range(num_convs, num_layers):
      if i != num_layers-1:
        self.params['W'+str(i+1)] = weight_scale*np.random.randn(input_before, hidden_dims[i-num_convs])
        self.params['b'+str(i+1)] = np.zeros((hidden_dims[i-num_convs],))
        input_before = hidden_dims[i-num_convs]
      else:
        self.params['W'+str(i+1)] = weight_scale*np.random.randn(input_before, num_classes)
        self.params['b'+str(i+1)] = np.zeros((num_classes))

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    filter_size = self.filter_size
    num_convs = self.num_convs
    num_layers = self.num_layers
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    
    scores = None
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    cache = []

    for i in xrange(num_convs):
      if i == 0:
        tmp_scores, tmp_cache = conv_relu_pool_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)], conv_param, pool_param)
      elif i == num_convs-1:
        tmp_scores, tmp_cache = conv_relu_forward(tmp_scores, self.params['W'+str(i+1)], self.params['b'+str(i+1)], conv_param)
      else:
        tmp_scores, tmp_cache = conv_relu_pool_forward(tmp_scores, self.params['W'+str(i+1)], self.params['b'+str(i+1)], conv_param, pool_param)
      
      cache.append(tmp_cache)

    for i in xrange(num_convs, num_layers):
      if i != num_layers-1:
        tmp_scores, tmp_cache = affine_relu_forward(tmp_scores, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
      else:
        scores, tmp_cache = affine_forward(tmp_scores, self.params['W'+str(i+1)], self.params['b'+str(i+1)])

      cache.append(tmp_cache)

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    for i in xrange(self.num_layers):
      W = self.params['W'+str(i+1)]
      loss += 0.5*self.reg*np.sum(W*W)
    

    for i in reversed(xrange(num_convs, num_layers)):
      if i == num_layers-1:
        dx, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_backward(dscores, cache[i])
      else:
        dx, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_relu_backward(dx, cache[i])

      grads['W'+str(i+1)] += self.reg*self.params['W'+str(i+1)]

    for i in reversed(xrange(num_convs)):
      if i == num_convs-1:
        dx, grads['W'+str(i+1)], grads['b'+str(i+1)] = conv_relu_backward(dx, cache[i])
      else:
        dx, grads['W'+str(i+1)], grads['b'+str(i+1)] = conv_relu_pool_backward(dx, cache[i])

      grads['W'+str(i+1)] += self.reg*self.params['W'+str(i+1)]
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
