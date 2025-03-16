from impl import LinearRegresion as LR, Plot2DUtils as PU
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y = x * (100 + np.random.randn(11) * 20) + 200 + np.random.randn(11) * 20
data = np.array([y, x])

parameters, errors = LR.linear_regression(data.T, 2, epochs = 100000, lr=0.0001)

print(parameters)
print(errors[0][-1], errors[1][-1])

figure, axis = PU.get_axis(2, 1)
PU.draw_points(axis[0], data)
PU.draw_line(axis[0], data, parameters)
PU.draw_points(axis[1], errors)
PU.show()