from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from impl import LinearRegresion as LR, Plot3DUtils as PU
import numpy as np

size = 20
x = np.linspace(0, 20, size)
y = np.linspace(0, 20, size)
x, y = np.meshgrid(x, y)
z = x * (np.random.randn(size, size) * 5 + 20) + y * (np.random.randn(size, size) * 4 + 10) + (np.random.randn(size, size) * 10 + 100)

data = np.array([ z.flatten(), y.flatten(), x.flatten() ])
parameters, errors = LR.linear_regression(data.T, 3, epochs = 100000, lr=0.00001)

print(parameters)
print(errors[0][-1], errors[1][-1])

axis_3d = PU.get_axis(121, '3d')
PU.draw_points(axis_3d, data)
PU.draw_surface(axis_3d, data, parameters)

axis_errors = PU.get_axis(122, None)
PU.draw_points_2d(axis_errors, errors)

PU.show()