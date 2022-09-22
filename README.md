# Deep Neural Network from Scratch
In this project, I build a deep neural network without the aid of any deep learning library (Tensorflow, Keras, Pytorch, …).
The reason for imposing myself on this task is that nowadays it is effortless to build deep and complex neural networks using the high-level tools provided by some python libraries. This approach allows machine learning practitioners to create powerful models with just a few lines of code, but it has the massive downside of leaving the functioning of those networks unclear.

This approach differs from other implementations for the strategy of storing the cached values. Also, differently from most implementation, this code allows to compare infinite possible network architectures as the number of layers and activation units is defined by the user.

An article describing in detail both the theoretical and coding part has been published on [Towards Data Science](https://towardsdatascience.com/). You can find it [here](https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a).

This repository is organized as follows.

`train.csv`
CSV file containing the training set.

`test.csv`
CSV file containing the test set.

`main.py`
Python script to:
- load the training and test sets
- set the network's architecture
- set the hyperparameters (learning step $\alpha$, number of iterations)
- launch the learning process
- save the learned weights and biases

`utils.py`
Python file to:
- shuffle data
- normalize data
- initialize parameters
- compute activation functions (ReLu, Softmax) and their derivatives
- One Hot Encode the data
- implement forward propagation
- implement back propagation
- update the network's parameters (weights and biases)
- compute cross entropy
- compute accuracy
- plot the learning curves

The script was tested on different network architectures:
![](https://github.com/andreoniriccardo/deep-neural-network-from-scratch/blob/main/images/training_acc_loss_%5B784%2C%2010%2C%2010%5D.png)
![](https://github.com/andreoniriccardo/deep-neural-network-from-scratch/blob/main/images/training_acc_loss_%5B784%2C%20256%2C%20128%2C%2064%2C%2010%5D.png)
