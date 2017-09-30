import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def linear_function():
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X), b)

    sess = tf.Session()
    result = sess.run(Y)
	
    sess.close()

    return result
	
def sigmoid(z):
    x = tf.placeholder(tf.float32, name = "x")
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict = {x: z})
    
    return result
	
def cost(logits, labels):
    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    
    sess = tf.Session()
    cost = sess.run(cost, feed_dict = {z: logits, y: labels})
    sess.close()
    
    return cost