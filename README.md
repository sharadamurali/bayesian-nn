# Bayesian Neural Networks (BNNs)

In a Bayesian Neural Network[1], the point-weights in a conventional model are replaced by probability distributions. This allows measuring uncertainities in estimations, thereby preventing the model from overfitting to the training data. However, the computational costs associated with modeling uncertainty for a vast number of parameters is extremely high, making this approach unfeasible for applications requiring large neural nets.

In this implementation, a BNN is used for the simple application of handwritten digit recognition, using the MNIST[2] dataset. The Tensorflow Probability[3] library has been
used in constructing this BNN, which has three fully-connected DenseFlipout[4] layers. In each layer, the kernel (analogous to layer weights) and the bias are assumed to be drawn from distributions, and the output is calculated similar to that in a traditional neural network:
```
kernel, bias ~ posterior
outputs = activation(matmul(inputs, kernel) + bias)
```

The Flipout estimator performs a Monte Carlo approximation of the distribution, integrating over the kernel and the bias. In this application, the estimator minimizes the Evidence Lower Bound (ELBO) loss. It consists of two terms: the expected negative log-likelihood (which is approximated via Monte Carlo), and the KL Divergence. This estimator uses about twice as many operations as a traditional neural network, which results in increased runtimes compared to a traditional fully-connected DNN.

## References

[1] Radford M. Neal. Bayesian Learning for Neural Networks. Springer-Verlag, 1996.

[2] The MNIST database. http://yann.lecun.com/exdb/mnist.

[3] Tensorflow probability. https://www.tensorflow.org/probability/overview.

[4] Tensorflow probability layers. https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/dense_variational.py.
