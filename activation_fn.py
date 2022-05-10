import numpy as np
# sigmoid function / log functiom
def sigmoid(z):
    a = np.zeros([1,1])
    a = 1 / (1 + np.exp(-z))

    return a

# derivitive of sigmoid function
def sigmoid_deriv(A):
    return A * (1 - A)

# tanh function
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

#derivative of tanh function
def tanh_derv(A):
    return 1 - (A ** 2)

# relu function
def relu(z):
    return np.maximum(0,z)

# derivitive of relu function
def relu_deriv(z):
    return np.greater(z, 0).astype(int)
    #return np.where(z <= 0, 0, 1)

def leaky_relu(x):
    return np.maximum(0.1 *x, x)