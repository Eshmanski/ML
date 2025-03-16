import matplotlib.pyplot as plt
import numpy as np

def get_axis(y, x):
    return plt.subplots(y, x)

def draw_points(axis, data):
    axis.plot(data[0], data[1], 'ob')

def draw_line(axis, data, parameters):
    padding = 5
    max_value = max(data[1])
    min_value = min(data[1])

    x = np.linspace(min_value - padding, max_value + padding, 10)
    y = x * parameters[1] + parameters[0]
    axis.plot(y, x, 'r-')

def show():
    plt.show()