import numpy as np
import matplotlib
from matplotlib import pyplot

def get_axis(y, x):
    return pyplot.subplots(y, x)

# Some functions to plot our points and draw the lines
def plot_points(axis, features, labels):
    X = np.array(features)
    y = np.array(labels)
    spam = X[np.argwhere(y == 1)]
    ham = X[np.argwhere(y == 0)]
    axis.scatter([s[0][0] for s in spam], [s[0][1] for s in spam], s=100, color='cyan', edgecolor='k', marker='^')
    axis.scatter([s[0][0] for s in ham], [s[0][1] for s in ham], s=100, color='red', edgecolor='k', marker='s')
    axis.set_xlabel('aack')
    axis.set_ylabel('beep')
    axis.legend(['happy', 'sad'])


def draw_line(axis, a, b, c, starting=0, ending=3, **kwargs):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    axis.plot(x, -c / b - a * x / b, **kwargs)

def draw_errors(axis, epochs, errors):
    axis.scatter(range(epochs), errors)

def show():
    pyplot.show()