import numpy as np
import random

def get_predicted(point, parameters):
    result = 0
    for i in range(len(parameters)):
        if i == 0:
            result += parameters[i]
        else:
            result += parameters[i] * point[i]
    return result

def get_difference(point, parameters):
    predicted = get_predicted(point, parameters)
    return point[0] - predicted

def init_parameters(dimension, factor):
    return np.array([ random.random() * factor for _ in range(dimension) ])

def get_new_parameters(point, parameters, lr = 0.001):
    value = point[0] - get_predicted(point, parameters)

    new_parameters = np.array([])
    for i in range(len(parameters)):
        if i == 0:
            new_parameters = np.append(new_parameters, parameters[i] + lr * value)
        else:
            new_parameters = np.append(new_parameters, parameters[i] + 2 * point[i] * lr * value)

    return new_parameters

def rmse(data, parameters):
    n = len(data)
    differences = np.array([get_difference(point, parameters) for point in data])
    return np.sqrt(1.0/n * (np.dot(differences, differences)))

def linear_regression(data, dimension, epochs = 100, lr = 0.001):
    p_count = len(data)
    parameters = init_parameters(dimension, 20)

    error = np.array([1, rmse(data, parameters)])
    errors = np.array([ error ])

    for epoch in range(epochs):
        i = random.randint(0, p_count - 1)
        parameters = get_new_parameters(data[i], parameters, lr)

        error = np.array([ epoch + 2, rmse(data, parameters) ])
        errors = np.append(errors, [error], axis=0)

    return parameters, errors.T