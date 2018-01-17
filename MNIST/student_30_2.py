from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
sess = tf.InteractiveSession()
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


soft_targets = np.load(file="soft-targets2.npy")


n_epochs = 50
batch_size = 50
num_nodes_h1 = 30 # Before 800
num_nodes_h2 = 30 # Before 800
learning_rate = 0.0001

n_batches = len(mnist.train.images) // batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def softmax_with_temperature(logits, temp=1.0, axis=1, name=None):
    logits_with_temp = logits / temp
    _softmax = tf.exp(logits_with_temp) / tf.reduce_sum(tf.exp(logits_with_temp), \
                                                        axis=axis, keep_dims=True)
    return _softmax

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
soft_target_ = tf.placeholder(tf.float32, [None, 10])

T = tf.placeholder(tf.float32)

W_h1 = weight_variable([784, num_nodes_h1])
b_h1 = bias_variable([num_nodes_h1])
h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)


W_h2 = weight_variable([num_nodes_h1, num_nodes_h2])
b_h2 = bias_variable([num_nodes_h2])
h2 = tf.nn.relu(tf.matmul(h1, W_h2) + b_h2)

W_output = tf.Variable(tf.zeros([num_nodes_h2, 10]))
b_output = tf.Variable(tf.zeros([10]))
logits = tf.matmul(h2, W_output) + b_output

y = tf.nn.softmax(logits)
y_soft_target = softmax_with_temperature(logits, temp=T)

loss_hard_target = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
loss_soft_target = -tf.reduce_sum(soft_target_ * tf.log(y_soft_target), \
                                  reduction_indices=[1])

loss = tf.reduce_mean(\
                      tf.square(T) * loss_hard_target \
                      + tf.square(T) * loss_soft_target)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


losses = []
accs = []
test_accs = []

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    x_shuffle, y_shuffle, soft_targets_shuffle \
            = shuffle(mnist.train.images, mnist.train.labels, soft_targets)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y, batch_soft_targets \
                = x_shuffle[start:end], y_shuffle[start:end], soft_targets_shuffle[start:end]
        sess.run(train_step, feed_dict={
                                        x: batch_x, y_: batch_y, soft_target_:batch_soft_targets, 
                                        T:2.0})
    train_loss = sess.run(loss, feed_dict={
                                        x: batch_x, y_: batch_y, soft_target_:batch_soft_targets, 
                                        T:2.0})
    train_accuracy = sess.run(accuracy, feed_dict={
                                        x: batch_x, y_: batch_y, T:1.0})
    test_accuracy = sess.run(accuracy, feed_dict={
                                        x: mnist.test.images, y_: mnist.test.labels, T:1.0})
    print("Epoch : %i, Loss : %f, Accuracy: %f, Test accuracy: %f" % (
            epoch+1, train_loss, train_accuracy, test_accuracy))
    losses.append(train_loss)
    accs.append(train_accuracy)
    test_accs.append(test_accuracy)

print("... completed!")



