import numpy as np

def init_params(layers_dims):
  params = {}
  for layer in range(1,len(layers_dims)):
    params['W'+str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer-1])*0.01
    params['b'+str(layer)] = np.random.randn(layers_dims[layer])*0.01

  return params
  
def relu(Z):
  return np.maximum(Z,0)

def sigmoid(Z):
  A = np.exp(Z)/sum(np.exp(Z))
  return A

def forward_prop(A_prev, W, b, activation):
  """
  Input: A_prev (b,c); W (a,b); b (a,1); activation (sigmoid or relu)
  Output: A, Z (a,c)
  """
  if activation == 'relu':
    Z = np.dot(W, A_prev) + b
    A = relu(Z)
  elif activation == 'sigmoid':
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
  else:
    raise Exception('Unknown activation function.')
  return A, Z

def one_hot(Y):
  """
  Y should have shape n,1 where n is the number of classes.
  Y comes in integer form (e.g. 4) and should be converted in binary shape:
  Y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]^T
  """
  # create temporary zeros array of shape (m,n), where m is the number
  # of training examples in Y, n is the number of classes in Y
  Y_zero = np.zeros((Y.shape[0], Y.max()+1))
  # set to 1 the corret indices
  Y_one_hot = Y_zero[np.arange(Y.shape[0]), Y] = 1
  # transpose
  Y_one_hot = Y_one_hot.T
  return Y_one_hot