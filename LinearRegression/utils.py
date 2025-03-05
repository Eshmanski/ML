import numpy as np
import matplotlib
from matplotlib import pyplot

def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8, opacity=0.2):
    x = np.linspace(starting, ending, 1000)
    pyplot.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth, alpha=opacity)

def plot_points(features, labels):
    x = np.array(features)
    y = np.array(labels)
    pyplot.scatter(x, y, color='blue', zorder=10)
    pyplot.xlabel('number of rooms')
    pyplot.ylabel('prices')
