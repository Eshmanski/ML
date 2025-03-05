import numpy as np

def simple_trick(base_price, price_per_room, num_rooms, price):
    small_random_1 = random.random()*0.1
    small_random_2 = random.random()*0.1
    predicted_price = base_price + price_per_room*num_rooms
    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random_1
        base_price += small_random_2
    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    if price < predicted_price and num_rooms > 0:
        price_per_room -= small_random_1
        base_price -= small_random_2
    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    return price_per_room, base_price

def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + price_per_room*num_rooms
    if price > predicted_price:
        price_per_room += learning_rate*num_rooms
        base_price += learning_rate
    else:
        price_per_room -= learning_rate*(num_rooms)
        base_price -= learning_rate
    return price_per_room, base_price

def square_trick(base, weight, x, y, learning_rate):
    pred_y = base + weight*x;
    new_weight = weight + 2 * learning_rate * x * (y - pred_y)
    new_base = base + 2 * learning_rate * (y - pred_y)
    return new_weight, new_base


def square_multy_trick(point, parameters, learning_rate):
    rest = point[1:]
    base = parameters[-1]
    weights = parameters[:-1]

    y = point[0]
    pred_y = np.dot(weights, rest) + base

    new_parameters = np.array([])
    for i in range(len(weights)):
        new_parameters = np.append(new_parameters, weights[i] + 2 * learning_rate * rest[i] * (y - pred_y))
    new_parameters = np.append(new_parameters, base + 2 * learning_rate * (y - pred_y))
    
    return new_parameters

