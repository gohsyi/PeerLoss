# learning-with-noisy-labels
Learning with Noisy Labels by adopting a peer prediction loss function.

## Requirements
* Python3
* Pytorch
* Pandas
* Numpy
* Scipy
* Sklearn

## Example
Run the following command in the terminal to replicate our experiment on UCI Heart Dataset.
```
python runner.py heart --seeds 8 --test-size 0.15 --val-size 0.1 --dropout 0 --loss bce --activation relu --normalize --verbose --e0 0.1 --e1 0.3 --episodes 1000 --batchsize 64 --batchsize-peer 64 --hidsize 8 --lr 0.0007 --alpha 1
```

If you want to equalize the prior by pre-sampling, add this argument: '--equalize-prior'.

### Details of the arguments:

* --dataset: name of the dataset, includes: 'heart', 'breast', 'breast2', 'german', 'banana', 'image', 'thyroid', 'titanic', 'splice', 'twonorm', 'waveform', 'flare-solar', 'diabetes', 'susy', 'higgs'
* --e0: error rate for class 0 (default: 0)
* --e1: error rate for class 1 (default: 0)
* --hidsize: size of hidden layers
* --lr: learning rate
* --batchsize: batchsize for training
* --batchsize-peer: batchsize for peer sampling
* --alpha: weight of peer term in peer loss
* --margin: margin for PAM
* --C1: weight of class 1 for C-SVM
* --dropout: dropout for neural network (deprecated, better without dropout)
* --activation: activation function, includes: relu, sigmoid, tanh, elu, relu6 (default: relu)
* --loss: loss function, includes: bce, mse, logistic, l1, huber (default: bce)
* --seeds: repeat experiments across how many seeds (default: 8)
* --episodes: training episodes
* --val-size: validation set proportion
* --test-size: test set proportion
* --equalize-prior: whether to equalize P(Y=1) and P(Y=0) (store_true)
* --normalize: whether to normalize the data
* --verbose: output more information (store_true)
