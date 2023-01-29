# scratch_neural_networks_MPI
Build neural network from scratch in python and parallelize it using MPI API 
### Overview

The project targets parallelize the neural network prediction, which is the dot product of the input with
the wights of each nodes to have the out of the node.
The problem is the algorithm of parallelizing that process depends on how we structure the network
wights.
For that I found that the best is to write the model by myself, so I can figure out how the model is
structured then parallelize the prediction.

For that you will see multiple files, some as library you can call them, some as implantation of those
files.

1- NNModel:
This file contains all the function , of building the network , then train it using Gradient descent
it contains too functions for two types of pruning, (structured and non structured)

2-NNModelPara
File for parallel predict function ( evaluation)

3-implantation :
In this file , we use the functions in sequential mode from NNModel, and plot data

4- implantat_para:
In this file , we use the functions in parallel mode from NNModelPara, and plot data

5- tenserflowtest : 
File to compare the models and the algorithms with tenserflow

6-networks_build: 
It is file where I used to build some networks to used them in implantat_para
