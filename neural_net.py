import numpy as np
from scipy.spatial.distance import euclidean

# Initialize parameters
def initialize_parameters(layers_dims):
  """
  Initialize parameters dictionary.
    
  Weight matrices will be initialized to random values from uniform normal
  distribution.
  bias vectors will be initialized to zeros.

  Arguments
  ---------
  layers_dims : list or array-like
      dimensions of each layer in the network.

  Returns
  -------
  parameters : dict
      weight matrix and the bias vector for each layer.
  """
  # np.random.seed(1)               
  parameters = {}
  L = len(layers_dims)

  for l in range(1, L):  
    r = np.sqrt(6) / np.sqrt(layers_dims[l] + layers_dims[l - 1] + 1)
    W = np.random.randn(layers_dims[l], layers_dims[l - 1])*2*r - r       
    parameters["W" + str(l)] = W
    parameters["b" + str(l)] = np.ones((layers_dims[l], 1))

    assert parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]), parameters["W" + str(l)].shape
    # assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

  return parameters

# Define activation functions that will be used in forward propagation
def sigmoid(Z):
  """
  Computes the sigmoid of Z element-wise.

  Arguments
  ---------
  Z : array
      output of affine transformation.

  Returns
  -------
  A : array
      post activation output.
  Z : array
      output of affine transformation.
  """
  A = 1 / (1 + np.exp(-Z))

  return A, Z


def tanh(Z):
  """
  Computes the Hyperbolic Tagent of Z elemnet-wise.

  Arguments
  ---------
  Z : array
      output of affine transformation.

  Returns
  -------
  A : array
      post activation output.
  Z : array
      output of affine transformation.
  """
  A = np.tanh(Z)

  return A, Z


def relu(Z):
  """
  Computes the Rectified Linear Unit (ReLU) element-wise.

  Arguments
  ---------
  Z : array
      output of affine transformation.

  Returns
  -------
  A : array
      post activation output.
  Z : array
      output of affine transformation.
  """
  A = np.maximum(0, Z)

  return A, Z


def leaky_relu(Z):
  """
  Computes Leaky Rectified Linear Unit element-wise.

  Arguments
  ---------
  Z : array
      output of affine transformation.

  Returns
  -------
  A : array
      post activation output.
  Z : array
      output of affine transformation.
  """
  A = np.maximum(0.1 * Z, Z)

  return A, Z

# Define helper functions that will be used in L-model forward prop
def linear_forward(A_prev, W, b):
  """
  Computes affine transformation of the input.

  Arguments
  ---------
  A_prev : 2d-array
      activations output from previous layer.
  W : 2d-array
      weight matrix, shape: size of current layer x size of previuos layer.
  b : 2d-array
      bias vector, shape: size of current layer x 1.

  Returns
  -------
  Z : 2d-array
      affine transformation output.
  cache : tuple
      stores A_prev, W, b to be used in backpropagation.
  """
  Z = np.dot(W, A_prev) + b
  cache = (A_prev, W, b)

  return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
  """
  Computes post-activation output using non-linear activation function.

  Arguments
  ---------
  A_prev : 2d-array
      activations output from previous layer.
  W : 2d-array
      weight matrix, shape: size of current layer x size of previuos layer.
  b : 2d-array
      bias vector, shape: size of current layer x 1.
  activation_fn : str
      non-linear activation function to be used: "sigmoid", "tanh", "relu".

  Returns
  -------
  A : 2d-array
      output of the activation function.
  cache : tuple
      stores linear_cache and activation_cache. ((A_prev, W, b), Z) to be used in backpropagation.
  """
  # assert activation_fn == "sigmoid" or activation_fn == "tanh" or \
  #     activation_fn == "relu"

  if activation_fn == "sigmoid":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = sigmoid(Z)

  elif activation_fn == "tanh":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = tanh(Z)

  elif activation_fn == "relu":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = relu(Z)

  elif activation_fn == "softmax":
      Z, linear_cache = linear_forward(A_prev, W, b)
      Zaux = Z - np.amax(Z,0)
      Zaux = np.exp(Zaux)
      A = Zaux/np.sum(Zaux,0)
      activation_cache = Zaux

  assert A.shape == (W.shape[0], A_prev.shape[1])

  cache = (linear_cache, activation_cache)

  return A, cache


def L_model_forward( X, parameters, hidden_layers_activation_fn="relu"):
  """
  Computes the output layer through looping over all units in topological
  order.

  Arguments
  ---------
  X : 2d-array
      input matrix of shape input_size x training_examples.
  parameters : dict
      contains all the weight matrices and bias vectors for all layers.
  hidden_layers_activation_fn : str
      activation function to be used on hidden layers: "tanh", "relu".

  Returns
  -------
  AL : 2d-array
      probability vector of shape 1 x training_examples.
  caches : list
      that contains L tuples where each layer has: A_prev, W, b, Z.
  """
  A = X
  caches = []                     
  L = len(parameters) //2    

  for l in range(1, L):
    A_prev = A
    A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation_fn=hidden_layers_activation_fn)
    caches.append(cache)

  AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation_fn="softmax")
  caches.append(cache)

  # assert AL.shape == (1, X.shape[1])

  return AL, caches

# Compute cross-entropy cost
def compute_cost(AL, y, W, numLayers, lambda_reg):
  """
  Computes the binary Cross-Entropy cost.

  Arguments
  ---------
  AL : 2d-array
      probability vector of shape 1 x training_examples.
  y : 2d-array
      true "label" vector.

  Returns
  -------
  cost : float
      binary cross-entropy cost.
  """
  m = y.shape[1]              
  # cost = - (1 / m)*np.sum(np.multiply(y, np.log(AL + 1e-30)) + np.multiply(1 - y, np.log(1 - AL)))
  cost = - (1 / m)*np.sum(np.multiply(y, np.log(AL + 1e-30)))
  regularization = 0
  for i in range(1,numLayers+1):
    regularization += np.sum(np.sum(np.square(W['W'+str(i)])))

  cost += (1 / m)*(lambda_reg/2)*regularization

  return cost

# Define derivative of activation functions w.r.t z that will be used in back-propagation
def sigmoid_gradient(dA, Z):
  """
  Computes the gradient of sigmoid output w.r.t input Z.

  Arguments
  ---------
  dA : 2d-array
      post-activation gradient, of any shape.
  Z : 2d-array
      input used for the activation fn on this layer.

  Returns
  -------
  dZ : 2d-array
      gradient of the cost with respect to Z.
  """
  A, Z = sigmoid(Z)
  dZ = dA * A * (1 - A)

  return dZ


def tanh_gradient(dA, Z):
  """
  Computes the gradient of hyperbolic tangent output w.r.t input Z.

  Arguments
  ---------
  dA : 2d-array
      post-activation gradient, of any shape.
  Z : 2d-array
      input used for the activation fn on this layer.

  Returns
  -------
  dZ : 2d-array
      gradient of the cost with respect to Z.
  """
  A, Z = tanh(Z)
  dZ = dA * (1 - np.square(A))

  return dZ


def relu_gradient(dA, Z):
  """
  Computes the gradient of ReLU output w.r.t input Z.

  Arguments
  ---------
  dA : 2d-array
      post-activation gradient, of any shape.
  Z : 2d-array
      input used for the activation fn on this layer.

  Returns
  -------
  dZ : 2d-array
      gradient of the cost with respect to Z.
  """
  A, Z = relu(Z)
  dZ = np.multiply(dA, np.int64(A > 0))

  return dZ


# define helper functions that will be used in L-model back-prop
def linear_backword(dZ, cache, reg):
  """
  Computes the gradient of the output w.r.t weight, bias, and post-activation
  output of (l - 1) layers at layer l.

  Arguments
  ---------
  dZ : 2d-array
      gradient of the cost w.r.t. the linear output (of current layer l).
  cache : tuple
      values of (A_prev, W, b) coming from the forward propagation in the current layer.

  Returns
  -------
  dA_prev : 2d-array
      gradient of the cost w.r.t. the activation (of the previous layer l-1).
  dW : 2d-array
      gradient of the cost w.r.t. W (current layer l).
  db : 2d-array
      gradient of the cost w.r.t. b (current layer l).
  """
  A_prev, W, b = cache
  m = A_prev.shape[1]

  dW = (1 / m) * np.dot(dZ, A_prev.T) + (1/m)*reg*W
  db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
  dA_prev = np.dot(W.T, dZ)

  assert dA_prev.shape == A_prev.shape
  assert dW.shape == W.shape
  assert db.shape == b.shape

  return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_fn, reg):
  """
  Arguments
  ---------
  dA : 2d-array
      post-activation gradient for current layer l.
  cache : tuple
      values of (linear_cache, activation_cache).
  activation : str
      activation used in this layer: "sigmoid", "tanh", or "relu".

  Returns
  -------
  dA_prev : 2d-array
      gradient of the cost w.r.t. the activation (of the previous layer l-1), same shape as A_prev.
  dW : 2d-array
      gradient of the cost w.r.t. W (current layer l), same shape as W.
  db : 2d-array
      gradient of the cost w.r.t. b (current layer l), same shape as b.
  """
  linear_cache, activation_cache = cache

  if activation_fn == "sigmoid":
      dZ = sigmoid_gradient(dA, activation_cache)
      dA_prev, dW, db = linear_backword(dZ, linear_cache, reg)

  elif activation_fn == "tanh":
      dZ = tanh_gradient(dA, activation_cache)
      dA_prev, dW, db = linear_backword(dZ, linear_cache, reg)

  elif activation_fn == "relu":
      dZ = relu_gradient(dA, activation_cache)
      dA_prev, dW, db = linear_backword(dZ, linear_cache, reg)

  return dA_prev, dW, db


def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu", reg=1e-2):
  """
  Computes the gradient of output layer w.r.t weights, biases, etc. starting
  on the output layer in reverse topological order.

  Arguments
  ---------
  AL : 2d-array
      probability vector, output of the forward propagation (L_model_forward()).
  y : 2d-array
      true "label" vector (containing 0 if non-cat, 1 if cat).
  caches : list
      list of caches for all layers.
  hidden_layers_activation_fn :
      activation function used on hidden layers: "tanh", "relu".

  Returns
  -------
  grads : dict
      with the gradients.
  """
  y = y.reshape(AL.shape)

  L = len(caches)
  grads = {}

  # dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

  # grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
  #     "db" + str(L)] = linear_activation_backward(
  #         dAL, caches[L - 1], "sigmoid", reg)

  # for l in range(L - 1, 0, -1):
  #     current_cache = caches[l - 1]
  #     grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
  #         "db" + str(l)] = linear_activation_backward(
  #             grads["dA" + str(l)], current_cache,
  #             hidden_layers_activation_fn, reg)

  m = y.shape[1]
  Wgradient = {}
  delta = -(1/m)*(y-AL)

  for l in range(L - 1, -1, -1):
      current_cache = caches[l]
      linear_cache, activation_cache = current_cache
      A_prev, W, b = linear_cache

      Wgradient[l] = np.hstack((b,W))

      if l==(L-1):
        Wgradient[l] = np.dot(delta,np.vstack((np.ones((1,A_prev.shape[1])),A_prev)).T)
      else:
        Wgradient[l] = np.dot(delta[1:,:],np.vstack((np.ones((1,A_prev.shape[1])),A_prev)).T)
      
      Wgradient[l][:,1:] += (1/m)*reg*W

      grads["dW"+str(l+1)] = Wgradient[l][:,1:]
      grads["db"+str(l+1)] = np.reshape(Wgradient[l][:,0],(Wgradient[l][:,0].shape[0],1))

      if l == 0:
        break
      
      current_cache = caches[l-1]
      linear_cache, activation_cache = current_cache
      Z = activation_cache

      if hidden_layers_activation_fn == "tanh":
        trans_func_der = 1 - np.square(2/(1+np.exp(-2*Z))-1)

      if l < L-1:
        delta = np.multiply(delta[1:,:].T@np.hstack((b,W)),np.vstack((np.ones((1,A_prev.shape[1])),trans_func_der)).T)
      else:
        delta = np.multiply(delta.T@np.hstack((b,W)),np.vstack((np.ones((1,A_prev.shape[1])),trans_func_der)).T)

      delta = delta.T


  return grads

# define the function to update both weight matrices and bias vectors
def update_parameters(parameters, grads, learning_rate):
  """
  Update the parameters' values using gradient descent rule.

  Arguments
  ---------
  parameters : dict
      contains all the weight matrices and bias vectors for all layers.
  grads : dict
      stores all gradients (output of L_model_backward).

  Returns
  -------
  parameters : dict
      updated parameters.
  """
  L = len(parameters) // 2

  for l in range(1, L + 1):
      parameters["W" + str(l)] = parameters[
          "W" + str(l)] - learning_rate * grads["dW" + str(l)]
      parameters["b" + str(l)] = parameters[
          "b" + str(l)] - learning_rate * grads["db" + str(l)]

  return parameters

# Define the multi-layer model using all the helper functions we wrote before


def L_layer_model(
      X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
      print_cost=False, hidden_layers_activation_fn="relu", lambda_reg=1e-2):
  """
  Implements multilayer neural network using gradient descent as the
  learning algorithm.

  Arguments
  ---------
  X : 2d-array
      data, shape: number of examples x num_px * num_px * 3.
  y : 2d-array
      true "label" vector, shape: 1 x number of examples.
  layers_dims : list
      input size and size of each layer, length: number of layers + 1.
  learning_rate : float
      learning rate of the gradient descent update rule.
  num_iterations : int
      number of iterations of the optimization loop.
  print_cost : bool
      if True, it prints the cost every 100 steps.
  hidden_layers_activation_fn : str
      activation function to be used on hidden layers: "tanh", "relu".

  Returns
  -------
  parameters : dict
      parameters learnt by the model. They can then be used to predict test examples.
  """
  # np.random.seed(1)

  # initialize parameters
  parameters = initialize_parameters(layers_dims)

  # intialize cost list
  cost_list = []

  ############MATLAB fmincg##############
  RHO   = 0.01                                                                     # a bunch of constants for line searches
  SIG   = 0.5                                                                      # RHO and SIG are the constants in the Wolfe-Powell conditions
  INT   = 0.1                                                                      # don't reevaluate within 0.1 of the limit of the current bracket
  EXT   = 3.0                                                                      # extrapolate maximum 3 times the current bracket
  MAX   = 20                                                                       # max 20 function evaluations per line search
  RATIO = 100                                                                      # maximum allowed slope ratio

  L = len(parameters) // 2

  red = 1
  S=['Iteration: ']
  df1_aux = np.empty(([]))
  for l in range(1, L + 1):
      df1_temp = np.hstack((parameters["b" + str(l)],parameters["W" + str(l)]))
      df1_aux = np.vstack((df1_aux,np.reshape(np.transpose(df1_temp),(np.shape(df1_temp)[0]*np.shape(df1_temp)[1],1))))
  W = df1_aux[1:]
  
  ####################Ler um W especifico de um .csv###############
  # path = "X.csv"
    
  # df = pd.read_csv(path, sep=r'\s*,\s*',header=None, encoding='ascii', engine='python')
  # df.columns = ["X1"]
  # W = df.values
  # SP = 0
  # for l in range(1, L + 1):
  #   EP = SP+np.shape(parameters["b" + str(l)])[0]
  #   parameters["b" + str(l)] = W[SP:EP]
  #   SP = EP
  #   EP = SP+np.shape(parameters["W" + str(l)])[0]*np.shape(parameters["W" + str(l)])[1]
  #   parameters["W" + str(l)] = np.transpose(np.reshape(W[SP:EP],(np.shape(parameters["W" + str(l)])[0],np.shape(parameters["W" + str(l)])[1])))
  #   SP = EP
  #############################################################
  
  # i = 0                                                                            # zero the run length counter
  ls_failed = 0                                                                    # no previous line search has failed
  fW = np.empty(([]))

  # iterate over L-layers to get the final output and the cache
  AL, caches = L_model_forward(
      X, parameters, hidden_layers_activation_fn)

  # compute cost to plot it
  cost = compute_cost(AL, y, parameters, len(parameters)//2, lambda_reg)

  # iterate over L-layers backward to get gradients
  grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn, lambda_reg)

  f1 = cost
  df1_aux = np.empty(([]))
  for l in range(1, L + 1):
      df1_temp = np.hstack((grads["db" + str(l)],grads["dW" + str(l)]))
      df1_aux = np.vstack((df1_aux,np.reshape(np.transpose(df1_temp),(np.shape(df1_temp)[0]*np.shape(df1_temp)[1],1))))
  df1 = df1_aux[1:]
  # print(f1)
  # print(df1)
  # input("Press Enter to continue...")
  # i = i + (length<0);                                                            # count epochs?!
  s = -df1                                                                         # search direction is steepest
  d1 = np.dot(-np.transpose(s),s)                                                                # this is the slope
  z1 = red/(1-d1)                                                                  # initial step is red/(|s|+1)

  # iterate over num_iterations
  for i in range(num_iterations):
      # # iterate over L-layers to get the final output and the cache
      # AL, caches = L_model_forward(
      #     X, parameters, hidden_layers_activation_fn)

      # # compute cost to plot it
      # cost = compute_cost(AL, y, parameters, len(parameters)//2, lambda_reg)
      # # iterate over L-layers backward to get gradients
      # grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn, lambda_reg)

      # # update parameters
      # parameters = update_parameters(parameters, grads, learning_rate)

      ############MATLAB fmincg#############
      W0 = W; f0 = f1; df0 = df1                                                 # make a copy of current values
      W = W + z1*s;                                                              # begin line search

      SP = 0
      for l in range(1, L + 1):
        EP = SP+np.shape(parameters["b" + str(l)])[0]
        parameters["b" + str(l)] = W[SP:EP]
        SP = EP
        EP = SP+np.shape(parameters["W" + str(l)])[0]*np.shape(parameters["W" + str(l)])[1]
        parameters["W" + str(l)] = np.transpose(np.reshape(W[SP:EP],(np.shape(parameters["W" + str(l)])[0],np.shape(parameters["W" + str(l)])[1])))
        SP = EP

      # iterate over L-layers to get the final output and the cache
      AL, caches = L_model_forward(
          X, parameters, hidden_layers_activation_fn)

      # compute cost to plot it
      cost = compute_cost(AL, y, parameters, len(parameters)//2, lambda_reg)

      # iterate over L-layers backward to get gradients
      grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn, lambda_reg)

      f2 = cost
      df1_aux = np.empty(([]))
      for l in range(1, L + 1):
          df1_temp = np.hstack((grads["db" + str(l)],grads["dW" + str(l)]))
          df1_aux = np.vstack((df1_aux,np.reshape(np.transpose(df1_temp),(np.shape(df1_temp)[0]*np.shape(df1_temp)[1],1))))
      df2 = df1_aux[1:]

      # i = i + (num_iterations<0);                                               # count epochs?!
      d2 = np.dot(np.transpose(df2),s);
      f3 = f1; d3 = d1; z3 = -z1;                                               # initialize point 3 equal to point 1
      if num_iterations>0:
        M = MAX
      else:
        M = min(MAX, -num_iterations-i)
      success = 0; limit = -1                                                   # initialize quanteties

      while True:
        while ((f2 > f1+z1*RHO*d1) or (d2 > -SIG*d1)) and (M > 0):
          limit = z1                                                            # tighten the bracket
          if f2 > f1:
            z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)                              # quadratic fit
          else:
            A = 6*(f2-f3)/z3+3*(d2+d3)                                          # cubic fit
            B = 3*(f3-f2)-z3*(d3+2*d2)
            z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A                                  # numerical error possible - ok!

          if np.isnan(z2) or np.isinf(z2):
            z2 = z3/2                                                           # if we had a numerical problem then bisect
          
          z2 = max(min(z2, INT*z3),(1-INT)*z3)                                  # don't accept too close to limits
          z1 = z1 + z2                                                          # update the step
          W = W + z2*s

          SP = 0
          for l in range(1, L + 1):
            EP = SP+np.shape(parameters["b" + str(l)])[0]
            parameters["b" + str(l)] = W[SP:EP]
            SP = EP
            EP = SP+np.shape(parameters["W" + str(l)])[0]*np.shape(parameters["W" + str(l)])[1]
            parameters["W" + str(l)] = np.transpose(np.reshape(W[SP:EP],(np.shape(parameters["W" + str(l)])[0],np.shape(parameters["W" + str(l)])[1])))
            SP = EP

          # iterate over L-layers to get the final output and the cache
          AL, caches = L_model_forward(
              X, parameters, hidden_layers_activation_fn)

          # compute cost to plot it
          cost = compute_cost(AL, y, parameters, len(parameters)//2, lambda_reg)

          # iterate over L-layers backward to get gradients
          grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn, lambda_reg)

          f2 = cost
          df1_aux = np.empty(([]))
          for l in range(1, L + 1):
              df1_temp = np.hstack((grads["db" + str(l)],grads["dW" + str(l)]))
              df1_aux = np.vstack((df1_aux,np.reshape(np.transpose(df1_temp),(np.shape(df1_temp)[0]*np.shape(df1_temp)[1],1))))
          df2 = df1_aux[1:]

          M = M - 1
          # i = i + (length<0);                           % count epochs?!
          d2 = np.dot(np.transpose(df2),s)
          z3 = z3-z2                                                            # z3 is now relative to the location of z2
        
        if f2 > f1+z1*RHO*d1 or d2 > -SIG*d1:
          break                                                                 # this is a failure
        elif d2 > SIG*d1:
          success = 1; break                                                    # success
        elif M == 0:
          break                                                                 # failure
        
        A = 6*(f2-f3)/z3+3*(d2+d3)                                              # make cubic extrapolation
        B = 3*(f3-f2)-z3*(d3+2*d2)
        z2 = -d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3))                              # num. error possible - ok!
        if ~np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2 < 0:            # num prob or wrong sign?
          if limit < -0.5:                                                      # if we have no upper limit
            z2 = z1 * (EXT-1)                                                   # the extrapolate the maximum amount
          else:
            z2 = (limit-z1)/2                                                   # otherwise bisect
        
        elif (limit > -0.5) and (z2+z1 > limit):                                # extraplation beyond max?
          z2 = (limit-z1)/2                                                     # bisect
        elif (limit < -0.5) and (z2+z1 > z1*EXT):                               # extrapolation beyond limit
          z2 = z1*(EXT-1.0)                                                     # set to extrapolation limit
        elif z2 < -z3*INT:
          z2 = -z3*INT
        elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-INT)):                    # too close to limit?
          z2 = (limit-z1)*(1.0-INT)
        
        f3 = f2; d3 = d2; z3 = -z2                                              # set point 3 equal to point 2
        z1 = z1 + z2
        W = W + z2*s                                                            # update current estimates

        SP = 0
        for l in range(1, L + 1):
          EP = SP+np.shape(parameters["b" + str(l)])[0]
          parameters["b" + str(l)] = W[SP:EP]
          SP = EP
          EP = SP+np.shape(parameters["W" + str(l)])[0]*np.shape(parameters["W" + str(l)])[1]
          parameters["W" + str(l)] = np.transpose(np.reshape(W[SP:EP],(np.shape(parameters["W" + str(l)])[0],np.shape(parameters["W" + str(l)])[1])))
          SP = EP

        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, y, parameters, len(parameters)//2, lambda_reg)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn, lambda_reg)

        f2 = cost
        df1_aux = np.empty(([]))
        for l in range(1, L + 1):
            df1_temp = np.hstack((grads["db" + str(l)],grads["dW" + str(l)]))
            df1_aux = np.vstack((df1_aux,np.reshape(np.transpose(df1_temp),(np.shape(df1_temp)[0]*np.shape(df1_temp)[1],1))))
        df2 = df1_aux[1:]
        M = M - 1
        # i = i + (length<0);                             % count epochs?!
        d2 = np.dot(np.transpose(df2),s)
        # end of line search

      if success:                                                               # if line search succeeded
        f1 = f2; fW = np.vstack((fW,f1))
        # print('%s %4i | Cost: %4.6e\r', S, i, f1)
        s = (np.dot(np.transpose(df2),df2)-np.dot(np.transpose(df1),df2))/(np.dot(np.transpose(df1),df1))*s - df2         # Polack-Ribiere direction
        tmp = df1; df1 = df2; df2 = tmp                                         # swap derivatives
        d2 = np.dot(np.transpose(df1),s)
        if d2 > 0:                                                              # new slope must be negative
          s = -df1                                                              # otherwise use steepest direction
          d2 = np.dot(-np.transpose(s),s)    
        
        z1 = z1 * min(RATIO, d1/(d2-np.finfo(float).tiny))                      # slope ratio but max RATIO
        d1 = d2
        ls_failed = 0                                                           # this line search did not fail
      else:
        W = W0; f1 = f0; df1 = df0                                              # restore point from before failed line search
        if ls_failed or i > abs(num_iterations):                                # line search failed twice in a row
          break                                                                 # or we ran out of time, so we give up
        
        tmp = df1; df1 = df2; df2 = tmp                                         # swap derivatives
        s = -df1                                                                # try steepest
        d1 = np.dot(-np.transpose(s),s)
        z1 = 1/(1-d1)                     
        ls_failed = 1                                                           # this line search failed

      # append each 100th cost to the cost list
      if (i + 1) % 100 == 0 and print_cost:
          print(f"The cost after {i + 1} iterations is: {cost:.4f}")

      if i % 100 == 0:
          cost_list.append(cost)

  # plot the cost curve
  # plt.figure(figsize=(10, 6))
  # plt.plot(cost_list)
  # plt.xlabel("Iterations (per hundreds)")
  # plt.ylabel("Loss")
  # plt.title(f"Loss curve for the learning rate = {learning_rate}")

  SP = 0
  for l in range(1, L + 1):
    EP = SP+np.shape(parameters["b" + str(l)])[0]
    parameters["b" + str(l)] = W[SP:EP]
    SP = EP
    EP = SP+np.shape(parameters["W" + str(l)])[0]*np.shape(parameters["W" + str(l)])[1]
    parameters["W" + str(l)] = np.transpose(np.reshape(W[SP:EP],(np.shape(parameters["W" + str(l)])[0],np.shape(parameters["W" + str(l)])[1])))
    SP = EP
  
  return parameters

def accuracy(X, parameters, y, activation_fn="relu"):
  """
  Computes the average accuracy rate.

  Arguments
  ---------
  X : 2d-array
      data, shape: number of examples x num_px * num_px * 3.
  parameters : dict
      learnt parameters.
  y : 2d-array
      true "label" vector, shape: 1 x number of examples.
  activation_fn : str
      activation function to be used on hidden layers: "tanh", "relu".

  Returns
  -------
  accuracy : float
      accuracy rate after applying parameters on the input data
  """
  probs, caches = L_model_forward(X, parameters, activation_fn)
  labels = (probs >= 0.5) * 1
  accuracy = np.mean(labels == y) * 100

  return f"The accuracy rate is: {accuracy:.2f}%."
