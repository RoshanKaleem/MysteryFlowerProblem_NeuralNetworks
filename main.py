from matplotlib import pyplot as plt
import numpy as np

data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

mystery_flower = [4.5, 1]

# print(data[2][1])

# network
w1 = np.random.randn()
print(w1)
w2 = np.random.randn()
print(w2)
b = np.random.randn()
print(b)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


T = np.linspace(-5, 5, 100)
Y = sigmoid_p(T)
plt.plot(T, Y)
Y = sigmoid(T)
plt.plot(T, Y)
plt.show()

# scatter data
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], color=color)
plt.grid()
plt.show()
# training
learning_rate = 0.2
costs = []
for i in range(1000000):
    randomnum = np.random.randint(len(data))
    point = data[randomnum]

    z = point[0] * w1 + point[1] * w2 + b
    h = sigmoid(z)

    target = point[2]
    cost = np.square(h - target)

    dcost_pred = 2 * (h - target)
    dpred_z = sigmoid_p(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_pred * dpred_z * dz_dw1
    dcost_dw2 = dcost_pred * dpred_z * dz_dw2
    dcost_db = dcost_pred * dpred_z * dz_db

    if i % 10000 == 0:
        print(cost)

    costs.append(cost)
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

plt.plot(costs)
plt.show()

print("predictions")
for i in range(len(data)):
    point = data[i]
    print(point)
    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)
    # print(prediction)


def guess_the_flower(length, width):
    z = length * w1 + width * w2 + b
    prediction = sigmoid(z)
    print(prediction)
    if prediction <= 0.5:
        print("Blue Flower")
    else:
        print("Red flower")


guess_the_flower(mystery_flower[0], mystery_flower[1])
