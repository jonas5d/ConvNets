from nolearn.lasagne import objective
from lasagne.layers import get_all_params
from lasagne import layers
from lasagne import nonlinearities
from lasagne import updates
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne import TrainSplit
import numpy as np
from custom_trainTest import CustomTrainSplit

def create_net(X_train,X_test,Y_train,Y_test,epochs=500,cuda_layers=True,):
  if cuda_layers:
    import lasagne.layers.cuda_convnet
    convLayer = lasagne.layers.cuda_convnet.Conv2DCCLayer
    poolLayer = lasagne.layers.cuda_convnet.MaxPool2DCCLayer
  else:
    convLayer = layers.Conv2DLayer
    poolLayer = layers.MaxPool2DLayer

  net = NeuralNet(
    layers=[ 
        ('input', layers.InputLayer),
        
        ('plotConv1', convLayer),

        ('pool1', poolLayer),
        
        ('dropout1', layers.DropoutLayer),
        
        ('conv2', convLayer),

        ('pool2', poolLayer),

        ('dropout2', layers.DropoutLayer),
        
        ('conv3', convLayer),
        
        ('pool3', poolLayer),

        ('dropout3', layers.DropoutLayer),
        
        ('hidden4', layers.DenseLayer),
        
        ('dropout4', layers.DropoutLayer),
        
        ('hidden5', layers.DenseLayer),
        
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, X_train.shape[1], X_train.shape[2],X_train.shape[3]),
    plotConv1_num_filters=32, plotConv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.3,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.5,  # !
    hidden4_num_units=1000,
    dropout4_p=0.5,  # !
    hidden5_num_units=1000,
    output_num_units=2, output_nonlinearity=nonlinearities.softmax,
    objective=regularization_objective,
    objective_lambda2=0.0025,
    update=updates.adam,
    update_learning_rate=0.0002,
    max_epochs=epochs,
    verbose=1,
    train_split=CustomTrainSplit(eval_size=0,cutoff=X_train.shape[0]),
    )
  return net
  #net.fit(X.astype(np.float32), Y.astype(np.int32))

def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
  # default loss
  losses = objective(layers, *args, **kwargs)
  # get the layers' weights, but only those that should be regularized
  # (i.e. not the biases)
  weights = get_all_params(layers[-1], regularizable=True)
  # sum of absolute weights for L1
  sum_abs_weights = sum([abs(w).sum() for w in weights])
  # sum of squared weights for L2
  sum_squared_weights = sum([(w ** 2).sum() for w in weights])
  # add weights to regular loss
  losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
  return losses


def train_net():
  print "stuff"