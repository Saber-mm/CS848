# Federated Learning: accuracy and convergence speed (PyTorch)

Experiments are on least squares, logistic regression, MNIST. 

The purpose of these experiments is to illustrate the effectiveness of the different federated learning algorithms and improving their speed.

## Requirments
Install the following packages 
* python3
* pytorch
* torchvision
* numpy
* tensorboardX
* pickle

## Data
* For least squares and logistic regression, the data and partition is automatically generated according to the scheme described in our report.
* For MNIST, download train and test datasets manually or they will be automatically downloaded from torchvision datasets, and partitioned automatically (see sampling.py).

## Experiments
* For codes and configurations regarding least squares and logistic regression, go to the "/convex" folder.
* For codes and configurations regarding MNIST, go to the "/nonconvex" folder.

## Output and Plot
* Outputs of the experiment will be stored in pickle or npz files. You can access the files and plot them afterwards by writing your own plotting script.
* Remember to modify the output file names in the code according to your demand (especially for convex experiments) so that if you run multiple process at once, the output files won't be overwritten.
