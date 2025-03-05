import matplotlib.pyplot as plt
import numpy as np
import random
import Tricks
import utils
import csv 

prices = np.array([])
areas = np.array([])
badrooms = np.array([])
random.seed(0) 

middlePrices = np.array([])
middleAreas = np.array([])

features = np.array([1,2,3,5,6,7])
labels = np.array([155, 197, 244, 356,407,448])

with open('./LinearRegression/Hyderabad.csv', 'r') as file:
    reader = csv.reader(file)
    line = 0
    for row in reader:
        if line != 0:
            prices = np.append(prices, float(row[0]))
            areas = np.append(areas, float(row[1]))
            badrooms = np.append(badrooms, float(row[3]))
        if line == 1001:
            break
        line += 1

    size = 20
    j = 0
    while j < len(prices)-1:
        middlePrices = np.append(middlePrices, np.median(prices[j:j+size]))
        middleAreas = np.append(middleAreas, np.median(areas[j:j+size]))
        j += size

def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * (np.dot(differences, differences)))

def linear_regression(data, learning_rate=0.01, epochs = 1000):
    errors = []
    demensions = len(data)
    # parameters = np.array([random.random() for _ in range(demensions)])
    parameters = np.array([15000, 1])
    point = np.array([data[i][0] for i in range(demensions)])
    
    for epoch in range(epochs):
        # predictions = parameters[0]+parameters[1]
        # errors.append(rmse(labels, predictions))
        utils.draw_line(parameters[0], parameters[1], 'gray', starting=0, ending=3000, opacity=0.1)
        i = random.randint(0, len(data)-1)
        point = np.array([data[k][i] for k in range(demensions)])
        new_parameters = Tricks.square_multy_trick(point, parameters, learning_rate)
        parameters = new_parameters

    utils.draw_line(parameters[0], parameters[1], 'blue', starting=0, ending=3000, opacity=1)
    # utils.plot_points(data[1], data[0])
    print('Параметры:')
    print(parameters)
    plt.show()


    # plt.scatter(range(len(errors)), errors)
    # plt.show() 
    
print(prices)
print(areas)
print(middlePrices)
print(middleAreas)
# utils.plot_points(areas, prices)
utils.plot_points(middleAreas, middlePrices)
linear_regression(np.array([middlePrices, middleAreas]), epochs=10000, learning_rate=0.0000001)
