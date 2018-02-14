import numpy as np
import math
import tensorflow as tf

'''
Author: @kaustubhhiware
'''

input_nodes = 784
hidden_nodes_1 = hidden_nodes_2 = hidden_nodes_3 = 256
output_nodes = 10

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def relu(x):
    return tf.maximum(x, 0)

def relu_(x):
    return (x > 0)*x

class NN(object):
    def __init__(self):

        self.W = []
        self.W.append(tf.Variable(tf.random_uniform([hidden_nodes_1, input_nodes], minval=-0.01, maxval=0.01, dtype=tf.float64, name='W1')))
        self.W.append(tf.Variable(tf.random_uniform([hidden_nodes_2, hidden_nodes_1], minval=-0.01, maxval=0.01, dtype=tf.float64, name='W2')))
        self.W.append(tf.Variable(tf.random_uniform([hidden_nodes_3, hidden_nodes_1], minval=-0.01, maxval=0.01, dtype=tf.float64, name='W3')))
        self.W.append(tf.Variable(tf.random_uniform([output_nodes, hidden_nodes_1], minval=-0.01, maxval=0.01, dtype=tf.float64, name='W4')))

        self.b = []
        self.b.append(tf.get_variable("b1", [hidden_nodes_1, 1], initializer=tf.zeros_initializer(), dtype=tf.float64))
        self.b.append(tf.get_variable("b2", [hidden_nodes_2, 1], initializer=tf.zeros_initializer(), dtype=tf.float64))
        self.b.append(tf.get_variable("b3", [hidden_nodes_3, 1], initializer=tf.zeros_initializer(), dtype=tf.float64))
        self.b.append(tf.get_variable("b4", [output_nodes, 1], initializer=tf.zeros_initializer(), dtype=tf.float64))

    def forward(self, x):
        z1 = tf.add(tf.matmul(self.W[0], x), self.b[0])
        f1 = relu(z1)
        z2 = tf.add(tf.matmul(self.W[1], f1), self.b[1])
        f2 = relu(z2)
        z3 = tf.add(tf.matmul(self.W[2], f2), self.b[2])
        f3 = relu(z3)
        z4 = tf.add(tf.matmul(self.W[3], f3), self.b[3])
        f4 = relu(z4)
        return f4

    def backward(self, x, y, f1, f2):
        d_z2 = f2.copy()
        d_z2[y] -= 1
        d_b2 = d_z2
        d_w2 = np.dot(d_z2, f1.T)
        d_f1 = np.dot(self.W[1].T, d_z2)

        d_z1 = (f1 > 0)*d_f1
        d_b1 = d_z1
        d_w1 = np.dot(d_z1, x.T)

        return (d_w1, d_b1, d_w2, d_b2)

    def hidden_layers(self, x):
        z1 = np.dot(self.W[0], x) + self.b[0]
        f1 = relu_(z1)
        z2 = np.dot(self.W[1], f1) + self.b[1]
        f2 = relu_(z2)
        z3 = np.dot(self.W[2], f2) + self.b[2]
        f3 = relu_(z3)
        return [f1, f2, f3]
