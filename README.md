# FeedFowardNetwork

A CNTK version Feed Foward Network



1. An abstract of Feed Foward Network:

  (1) The input data is of m dimension, and it can be represent using m nodes.

  (2) m nodes then connected to n nodes in hidden layer, where n is the dimension of hidden layers.

  (3) The connection between Input Data & Hidden layer can be represented by a n * m Matrix

  (4) The connection between Hidden layers can be represented as n * n Matrix.

  (5) Add unlinearity after getting the 'feature' in hidden layers.

  (6) Fully connected layer is a Linear Layer of size(n * output_classes)

  

2. CNTK routine:

  (1) Create the model : define number of layers; dimensions; input dim; output classes; non-linearity(RELU, sigmod, etc)

  (2) Define Error : softmax..

  (3) Configure Trainning process : configure minibatch, chose training method(sgd)

  (4) Monitor & Evaluate : Print the error & result evaluation during training process..It seems that CNTK don't have good monitor utils...
  
 
