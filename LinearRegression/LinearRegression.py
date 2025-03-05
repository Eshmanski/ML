import matplotlib.pyplot as plt
import numpy as np
import Tricks
import random
import utils

random.seed(0)
features = np.array([1,2,3,5,6,7])
labels = np.array([155, 197, 244, 356,407,448])

# The root mean square error function
def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * (np.dot(differences, differences)))


def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    price_per_room = random.random()
    base_price = random.random()
    errors = []

    for epoch in range(epochs):
        # Uncomment any of the following lines to plot different epochs
        #if epoch == 1:
        #if epoch <= 10:
        #if epoch <= 50:
        # if epoch > 50:
        # if True:
        #     utils.draw_line(price_per_room, base_price, starting=0, ending=8)

        predictions = features[0]*price_per_room+base_price
        errors.append(rmse(labels, predictions))
        i = random.randint(0, len(features)-1)
        num_rooms = features[i]
        price = labels[i]

        # Uncomment any of the 2 following lines to use a different trick
        #price_per_room, base_price = simple_trick(base_price, price_per_room, num_rooms, price)
        #price_per_room, base_price = absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate)
        price_per_room, base_price = Tricks.square_trick(base_price, price_per_room, num_rooms, price, learning_rate)

    print('Price per room:', price_per_room)
    print('Base price:', base_price)

    plt.figure(1)
    plt.subplot(211)
    utils.draw_line(price_per_room, base_price, 'blue', starting=0, ending=8, opacity=1)
    utils.plot_points(features, labels)

    plt.subplot(212)
    plt.scatter(range(len(errors)), errors)

    plt.show()
    return price_per_room, base_price

# This line is for the x-axis to appear in the figure

linear_regression(features, labels, learning_rate = 0.01, epochs = 10000)
