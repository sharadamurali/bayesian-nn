# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:44:05 2019

@author: sharada murali

Bayesian NN for MNIST digit classification
"""

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import timeit

tfd = tfp.distributions

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10  # output layer (0-9 digits)

n_iterations = 1000
batch_size = 128
dropout = 0.5

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

# Define model
model = tf.keras.Sequential([
      tfp.layers.DenseFlipout(n_hidden1, activation=tf.nn.relu),
      tfp.layers.DenseFlipout(n_hidden2, activation=tf.nn.relu),
      tfp.layers.DenseFlipout(n_hidden3, activation=tf.nn.relu),
      tfp.layers.DenseFlipout(n_output),
  ])

logits = model(X)

# Compute ELBO loss, averaged.
neg_log_likelihood = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
kl = sum(model.losses) / n_train
elbo_loss = neg_log_likelihood + kl

# Optimizer
train_op = tf.train.AdamOptimizer().minimize(elbo_loss)

# Compute prediction accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# To keep track of loss and accuracy
losses = []
accuracies = []
iter_no = []

start = timeit.default_timer()
# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_op, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    if i % 10 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [elbo_loss, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        
        losses.append(minibatch_loss)
        accuracies.append(minibatch_accuracy)
        iter_no.append(i)
        if i % 100 == 0:
            print(
                "Iteration",
                str(i),
                "\t| Loss =",
                str(minibatch_loss),
                "\t| Accuracy =",
                str(minibatch_accuracy)
                )

stop = timeit.default_timer()

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)
print('Time: ', stop - start)

# Plot loss and accuracy
plt.figure()
plt.plot(iter_no, losses)
plt.title("Loss on Validation Set")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(iter_no, accuracies)
plt.title("Accuracy on Validation Set")
plt.xlabel('Iteration')
plt.ylabel('Accuracy %')
plt.show()