from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

tf.set_random_seed(1)

num_nodes = 1200


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

W_fc1 = weight_variable([784, num_nodes])
b_fc1 = bias_variable([num_nodes])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([num_nodes, num_nodes])
b_fc2 = bias_variable([num_nodes])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([num_nodes, 10])
b_fc3 = bias_variable([10])
logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i % 500 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x: batch[0], y_: batch[1], keep_prob: 1.0})
    print('step %d, training accuracy %g' % (i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

count.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})

def softmax_with_temperature(logits, temp=1.0, axis=1, name=None):
  logits_with_temp = logits / temp
  _softmax = tf.exp(logits_with_temp) / tf.reduce_sum(tf.exp(logits_with_temp), axis=axis, keep_dims=True)
  return _softmax

y_soft_target = softmax_with_temperature(logits, temp = 8.0)


_soft_targets = []
for i in range(1100):
  start = i*50
  end = start+50
  batch_x = mnist.train.images[start:end]
  soft_target = sess.run(y_soft_target, feed_dict = {x: batch_x, keep_prob: 1.0})
  _soft_targets.append(soft_target)


soft_targets  = np.c_[_soft_targets].reshape(55000, 10)

np.save('soft-targets8.npy', soft_targets)



