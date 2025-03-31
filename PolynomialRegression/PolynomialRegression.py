import matplotlib.pyplot as plt
import turicreate as tc
import numpy as np
import random

random.seed(0)

params = [15, 1, -1]

def polynomial(params, x):
    n = len(params)
    return sum([params[i]*x**i for i in range(n)])

def draw_polynomial(params):
    x = np.linspace(-5, 5, 100)
    y = np.array([polynomial(params, xi) for xi in x])

    plt.ylim(-20, 20)
    plt.plot(x, y, linestyle='-', color='black')



def display_results(model):
    params = model.coefficients
    print("Training error (rmse):", model.evaluate(train)['rmse'])
    print("Testing error (rmse):", model.evaluate(test)['rmse'])
    plt.scatter(train['x'], train['y'], marker='o')
    plt.scatter(test['x'], test['y'], marker='^')
    draw_polynomial(params['value'])
    print("Polynomial coefficients")
    print(params['name', 'value'])

X = []
Y = []
for i in range(40):
    x = random.uniform(-5,5)
    y = polynomial(params, x) + random.gauss(0,2)
    X.append(x)
    Y.append(y)

data = tc.SFrame({'x': X, 'y': Y})
for i in range(2,200):
    string = 'x^'+str(i)
    data[string] = data['x'].apply(lambda x:x**i)

train, test = data.random_split(.8, seed=0)

# model_no_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.0, verbose=False, validation_set=None)
# display_results(model_no_reg)

# model_L1_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.1, l2_penalty=0.0, verbose=False, validation_set=None)
# display_results(model_L1_reg)

model_L2_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.1, verbose=False, validation_set=None)
display_results(model_L2_reg)

# draw_polynomial(params)
# plt.scatter(X, Y)
plt.show()