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

def one_hot(Y):
  """
  Y should have shape n,1 where n is the number of classes.
  Y comes in integer form (e.g. 4) and should be converted in binary shape:
  Y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]^T
  """
  # create temporary zeros array of shape (m,n), where m is the number
  # of training examples in Y, n is the number of classes in Y
  Y_one_hot = np.zeros((Y.shape[0], Y.max()+1))
  # set to 1 the corret indices
  Y_one_hot[np.arange(Y.shape[0]), Y] = 1
  # transpose
  Y_one_hot = Y_one_hot.T
  return Y_one_hot

def forward_prop(X, params):
  """
  Forward propagation for the L layers.
  First (L-1) layers: relu activation
  Last layer: sigmoid activation
  """
  # number of layers (note: params contains W and b for each layer, so it's necessary to do //2)
  L = len(params) // 2
  
  activations = {}
  activations['A0'] = X

  # for layers 1 to L-1 apply relu activation
  for l in range(1,L):
    activations['Z'+str(l)] = np.dot(params['W'+str(l)], activations['A'+str(l-1)]) + params['b'+str(l)]
    activations['A'+str(l)] = relu(activations['Z'+str(l)])

  activations['Z'+str(L)] = np.dot(params['W'+str(L)], activations['A'+str(L-1)]) + params['b'+str(L)]
  activations['A'+str(L)] = sigmoid(activations['Z'+str(L)])
    
  
  
  return activations



  



def back_prop(activations, params, Y):
  """
  Inputs:
  activations: dictionary like {'A0':..., 'A1':..., 'Z1':..., 'A2':..., ...}
  params: dictionary like {'W1':..., 'b1':..., 'W2':...}
  Output:
  gra
  
  """
  m = Y.shape[1]
  L = len(params) // 2

  grads = {}
  # for last layer L
  one_hot_Y = one_hot(Y)
  dZ_l = activations['A'+str(L)] - one_hot_Y
  grads['dW'+str(L)] = 1 / m * np.dot(dZ_l, activations['A'+str(L-1)].T)
  grads['db'+str(L)] = 1 / m * np.sum(dZ_l)

  # for layers L-1 to 1
  for l in range(1, L):
    dZ_l = np.dot(params['W'+str(l+1)].T, dZ_l) * deriv_relu(cache['Z'+str(l)])
    grads['dW'+str(l)] = 1 / m * np.dot(dZ_l, activations['A'+str(l-1)].T)
    # NOTA MIA cache deve contenere cache = {'Z1':... , 'Z2': ...}
    # NOTA MIA  A0 = X
    grads['db'+str(l)] = 1 / m * np.sum(dZ_l)
  return grads

def update_params(params, grads, alpha):
  # number of layers
  L = len(params) // 2

  params_updated = {}
  for l in range(1, L+1):
    params_updated['W'+str(l)] = params['W'+str(l)] - alpha*grads['dW'+str(l)]
    params_updated['b'+str(l)] = params['b'+str(l)] - alpha*grads['db'+str(l)]

  return params_updated