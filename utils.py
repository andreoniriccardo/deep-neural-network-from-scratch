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
  if activation == 'relu':
    Z = np.dot(W, A_prev) + b
    A = relu(Z)
  elif activation == 'sigmoid':
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
  else:
    raise Exception('Unknown activation function.')
  return A, Z