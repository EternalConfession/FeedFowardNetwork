# Any learning program contains : 
# (1) Data Reading
# (2) Data Processing
# (3) Creating Model
# (4) Learning the Model Parameters
# (5) Testing

# Here we write a feedfoward Network, which is the first & simplest type artificial neural network.

from __future__ import print_function
import matplotlib.pyplot as pyplot
import numpy as np 
import sys
import os
import cntk as C
from cntk import Trainer, learning_rate_schedule, UnitType, sgd
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Dense

#Set the default device as GPU..
C.try_set_default_device(C.device.gpu(0))

#Generate Data...This could be viewed as collecting(processing) data
np.random.seed(0)
input_dim = 2
output_classes = 2

def generateData(sampleSize, input_dim, num_output_classes):
    Y = np.random.randint(low = 0, high = num_output_classes, size = (sampleSize,1))
    X = (np.random.randn(sampleSize, input_dim) + 3) * (Y + 1) 
    Y1 = [Y == class_num for class_num in range(num_output_classes)]
    Y = np.asarray(np.hstack(Y1), dtype=np.float32)
    return X,Y

#mySampleSize = 64
#features, labels = generateData(64, input_dim, output_classes)

#Create Model...
#We create the simplest feed foward network with 2 hidden layers...each have 50 hidden nodes.

num_hidden_layers = 2
num_layers_dim = 50

#inpu is a key container for our data

feature = C.input(input_dim)
label = C.input(output_classes)

#define a linear layer
def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    
    weight = C.parameter(shape = (input_dim, output_dim))
    bias = C.parameter(shape = (output_dim))

    return bias + C.times(input_var, weight)

#define a nonlinearity
def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)
    nonlinearity(l)

    return l

#Define the network
#We can define as follows, taking two layers as example:
#h1 = dense_layer(input_var, num_layers_dim, sigmod)
#h2 = dense_layer(h2, num_layers_dim, sigmod)
#A general way
#h = dense_layer(input_var, hidden_layer_dim, sigmod)
#for i in range(1, num_hidden_layers):
#    h = dense_layer(h, hidden_layer_dim, sigmod)


#Define the fully connected layer
def fully_connected_layer(input_var, output_dim, num_hiddden_layers, layer_dim, nonlinearity):
    h = dense_layer(input_var, layer_dim, nonlinearity)
    for i in range(1,num_hiddden_layers):
        h = dense_layer(h, layer_dim, nonlinearity)
    
    return linear_layer(h, output_dim)

z1 = fully_connected_layer(feature, output_classes, num_hidden_layers, num_layers_dim, C.sigmoid)

#Here we write the layers by hand...lol. We can use the layer_library instead...

#An example of use layer_library
def create_model(features):
    with default_options(init=glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = Dense(num_layers_dim)(h)
        last_layer = Dense(output_classes, activation = None)
        
        return last_layer(h)    
z2 = create_model(feature)

#Training
loss = C.cross_entropy_with_softmax(z2, label)
eval_error = C.classification_error(z2, label)

learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = C.sgd(z2.parameters, lr_schedule)
Trainer = Trainer(z2, (loss, eval_error), [learner])

#Utils function to monitor the training loss
def moving_average(a, w=10):    
    if len(a) < w: 
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):    
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

minibatch_size = 40
num_samples = 20000
num_minibatch = num_samples/minibatch_size

training_progress_output_freq = 20
plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatch)):
    features, labels = generateData(minibatch_size, input_dim, output_classes)

    Trainer.train_minibatch({feature: features, label: labels})
    batchsize, loss, error = print_training_progress(Trainer, i, 
                                                     training_progress_output_freq, verbose=0)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

#Compare the loss & error
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()