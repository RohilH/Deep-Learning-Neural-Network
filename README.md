# Deep-Learning-Neural-Network
Created a four layer Neural Network that works on the MNIST Fashion dataset.
This code builds a 4-layer neural network with 256 hidden nodes per layer except 
the last layer which has 10 (number of classes) layers. I use Minibatch Gradient Descent, which runs for a given number of epochs
and does the following per epoch:
Shuffle the training data
Split the data into batches (use batch size of 200)
For each batch (subset of data):
feed batch into the 4-layer neural network
Compute loss and update weights
Observe the total loss and go to next iteration

The code can be run by running python nn_main.py
