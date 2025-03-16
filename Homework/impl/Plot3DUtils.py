from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import alpha

fig = plt.figure()

def get_axis(num, projection = '3d'):
    return fig.add_subplot(num, projection=projection)

def draw_points(axis, data):
    axis.scatter(data[0], data[1], data[2], color='red')

def draw_points_2d(axis, data):
    axis.plot(data[0], data[1], 'ob')

def draw_surface(axis, data, parameters):
    padding = 10
    y_min = min(data[1])
    y_max = max(data[1])
    x_min = min(data[2])
    x_max = max(data[2])

    x = np.linspace(x_min - padding, x_max + padding, 10)
    y = np.linspace(y_min - padding, y_max + padding, 10)
    x, y = np.meshgrid(x, y)
    z = parameters[0] + y * parameters[1] + x * parameters[2]

    axis.plot_wireframe(z, y, x, alpha=0.2)

def show():
    plt.show()