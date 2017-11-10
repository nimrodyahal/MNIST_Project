import numpy as np


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


inputs = np.array([[3., 5.],
                  [5., 1.],
                  [10., 2.]])

output = np.array([[75.],
                   [82.],
                   [93.]])

inputs /= np.amax(inputs, axis=0)
output /= 100
print inputs
print output

np.random.seed(1)

syn0 = 2 * np.random.random((inputs[0].size, 4)) - 2
syn1 = 2 * np.random.random((4, output[0].size)) - 2

for j in range(60000):
    layer0 = inputs
    layer1 = nonlin(np.dot(layer0, syn0))
    layer2 = nonlin(np.dot(layer1, syn1))

    l2_error = output - layer2
    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(layer2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(layer1, deriv=True)

    syn1 += layer1.T.dot(l2_delta)
    syn0 += layer0.T.dot(l1_delta)

print("Output after training")
print(layer2)

new_input = raw_input()
inputs = np.array([map(float, new_input.split(' '))])
layer1 = nonlin(np.dot(inputs, syn0))
layer2 = nonlin(np.dot(layer1, syn1))
print layer2
