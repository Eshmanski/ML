from matplotlib import pyplot as plt
import numpy as np
import random
import utils

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
labels = np.array([0,0,0,0,1,1,1,1])

utils.plot_points(features, labels)

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def score(weights, bias, features):
    return np.dot(weights, features) + bias

def prediction(weights, bias, features):
    return sigmoid(score(weights, bias, features))

def log_loss(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    return -label * np.log(pred) - (1 - label * np.log(1 - pred))

def total_log_loss(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += log_loss(weights, bias, features[i], labels[i])
    return total_error

def logistic_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label - pred) * features[i] * learning_rate
    bias += (label - pred) * learning_rate
    return weights, bias

def logistic_regression_algorithm(features, labels, learning_rate = 0.01, epochs = 1000):
    utils.plot_points(features, labels)
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features) - 1)
        weights, bias = logistic_trick(weights, bias, features[j], labels[j], learning_rate)
        print(errors[i])

    return weights, bias

weights, bias = logistic_regression_algorithm(features, labels, 0.01, 3000)

print(weights, bias)
utils.draw_line(weights[0], weights[1], bias)
plt.show()