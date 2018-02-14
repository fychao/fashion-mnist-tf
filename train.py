import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import random_ops
import math
import os
import urllib
import requests
import pickle
from sklearn.linear_model import LogisticRegression
import data_loader
import module

'''
Author: @kaustubhhiware
run with python2
'''

'''Implement mini-batch SGD here'''
'''
python train.py --train: Train neural net and dump weights
python train.py --train --iter 5: Specify number of iterations for training
python train.py --test: Test with stored weights
python train.py --layer=1: Run logistic regression on intermediate results of layers and print accuracy
'''

# Network parameters
hidden_nodes_1 = hidden_nodes_2 = hidden_nodes_3 = 256
input_nodes = 784
output_nodes = 10
data = data_loader.DataLoader()
weights_folder = '/weights/'

# check if needed files are present or not. Downloads if needed.
def check_download_weights():
    
    url_prefix = 'https://raw.githubusercontent.com/kaustubhhiware/fashion-mnist-tf/master'
    files = ['checkpoint', 'model', 'model.ckpt.data-00000-of-00001','model.ckpt.index', 'model.ckpt.meta']

    for file in files:
        if not os.path.exists(os.getcwd() + weights_folder + file):
            print 'Downloading', file
            url = url_prefix + weights_folder + file
            # urllib.urlretrieve(url, filename= os.getcwd() + weights_folder + file) 
            r = requests.get(url)
            open(os.getcwd() + weights_folder + file, 'wb').write(r.content)
                   

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)

    logits_scaled = logits - tf.reduce_max(logits, [1], keepdims=True)
    logits_norm = logits_scaled - \
        tf.reduce_logsumexp(logits_scaled, [1], keepdims=True)

    return tf.reduce_mean(-tf.reduce_sum(labels * logits_norm, reduction_indices=[1]))


def model(train, test, layers, alpha=0.005, isTrain=False, num_iterations=50, batch_size=50):
    ops.reset_default_graph()
    seed = 123
    tf.set_random_seed(seed)
    np.random.seed(seed)
    (trainX, trainY) = train
    (testX, testY) = test
    (n_x, m) = trainX.T.shape
    neural_net = module.NN()
    costs = []  
    layers -= 1

    X = tf.placeholder(tf.float64, [input_nodes, None], name='X')
    Y = tf.placeholder(tf.float64, [output_nodes, None], name='Y')

    Z = neural_net.forward(X)
    cost = compute_cost(Z, Y)
    saver = tf.train.Saver()
    
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    parameters = {'W': neural_net.W, 'b': neural_net.b}
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if not os.path.isdir(os.getcwd() + weights_folder):
        print 'Missing folder made'
        os.makedirs(os.getcwd() + weights_folder)

    if isTrain:
        for iteration in range(num_iterations):
            iter_cost = 0.
            batch_x, batch_y = data.create_batches(trainX, trainY, batch_size=batch_size)
            num_minibatches = int(m / batch_size)

            for (minibatch_X, minibatch_Y) in zip(batch_x, batch_y):
                minibatch_X , minibatch_Y = np.asmatrix(minibatch_X), np.asmatrix(minibatch_Y)
                _, minibatch_cost, acc = sess.run([optimizer, cost, accuracy], feed_dict={ X: minibatch_X.T, Y: minibatch_Y.T})
                iter_cost += minibatch_cost*1.0 / num_minibatches
        
            print "Iteration {iter_num}, Cost: {cost}, Accuracy: {accuracy}".format(iter_num=iteration, cost=iter_cost, accuracy = acc)
            costs.append(iter_cost)
        parameters = {'W': neural_net.W, 'b': neural_net.b}
        parameters = sess.run(parameters)

        save_path = saver.save(sess, "." + weights_folder + "model.ckpt")
        with open('.' + weights_folder + 'model','w') as f:
            pickle.dump(parameters, f)
        print "Parameters have been trained and saved!"
        print "Train Accuracy:", accuracy.eval({X: trainX.T, Y: trainY.T}, session=sess)

    else:
        check_download_weights()
        if layers == -1: # testing loop
            saver.restore(sess, "." + weights_folder + "model.ckpt")
            neural_net.W = parameters['W']
            neural_net.b = parameters['b']
            print "Test Accuracy:", accuracy.eval({X: testX.T, Y: testY.T}, session=sess)
      
        else:        
            with open('.' + weights_folder + 'model', 'r') as f:
                temp = pickle.load(f)
            neural_net.W = temp['W']
            neural_net.b = temp['b']
            trainY, testY = np.argmax(trainY, axis=1), np.argmax(testY, axis=1)

            trainH = [neural_net.hidden_layers(trainX[i].T.reshape(-1, 1)) for i in range(trainX.shape[0])]
            activationTrainH = [trainH[i][layers].ravel() for i in range(len(trainH))]
            testH = [neural_net.hidden_layers(testX[i].T.reshape(-1, 1)) for i in range(testX.shape[0])]
            activationTestH = [testH[i][layers].ravel() for i in range(len(testH))]

            print "Logistic Regression on layer", layers+1
            LR = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            LR.fit(activationTrainH, trainY)
            predictions = LR.predict(activationTestH)
            accuracy = np.mean((predictions == testY))
            print "LR Accuracy", accuracy

    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action="store_true", help="Initiate training phase and store weights")
    parser.add_argument('--test',action="store_true", help="Initiate testing phase, load model and print accuracy")
    parser.add_argument('--layer', action="store", dest="layer", type=int, choices=[1, 2, 3], help="Logistic Regression results for layers")
    parser.add_argument('--iter', action="store", dest="iter", type=int, help="Specify number of iterations")
    
    trainX, trainY = data.load_data('train')
    train = (trainX, trainY)
    testX, testY = data.load_data('test')
    test = (testX, testY)
    isTrain_ = False
    layers = 0
    num_iterations_ = 50
    args = parser.parse_args()
    if args.layer:
        print "> layers flag has set value", args.layer
        layers = args.layer
    elif args.train:
        print "> Now Training"
        isTrain_ = True
        if args.iter:
            num_iterations_ = args.iter
    elif args.test:
        print "> Now Testing"
    else:
        print "> Need to provide train / test / layers flag!"
        exit(0)  

    model(train, test, layers, isTrain=isTrain_, num_iterations=num_iterations_)


if __name__ == '__main__':
    main()
