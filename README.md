# banknote-authentication-python-code
#Importing libraries 
import pandas as pd
import numpy as np
import time
from scipy import special
#Loading data and visualizing the dataset 
df = pd.read_csv('data_banknote_authentication.txt')
df.head(5)
y = np.array(df['Class'])
print("y shape: ", y.shape)
X = np.array(df.iloc[:, :-1])
from matplotlib import pyplot as plt
from numpy import where
for i in range(2):
	samples_ix = where(y == i)
	plt.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
plt.legend()
plt.show()
# Transposing our Features for later use
X = X.T
print(X.shape)

print(X)
print(y)
#Making hyperparamets configurable 
# configuring hyperparameters

def process_input(prompt, type, list=[]):
    while True:
        if(type == 'int'):
            try:
                value = int(input(prompt))
                break
            except ValueError:
                print("Please enter only integer")
                continue
        else:
            if list:
                print(list)
                value = input(prompt).lower()
                print("value = ", value)
                if value not in list:
                    print("Please enter only one of these: {}".format(list))
                    continue
                else:
                    break
            else:
                value = input(prompt)
                break
    return value
    # 
hidden_layer_len = process_input("Please enter number of hidden layers:", 'int')

hidden_layers = []
for i in range(int(hidden_layer_len)):
    neuron_len = process_input("Enter number of neurons in hidden layer: h{}".format(i+1), 'int')
    hidden_layers.append(neuron_len)

activation_fn_hidd = process_input("Please choose activation function for the hidden layer"+
                            "\nAvailable Activations: [sigmoid, relu, tanh]", 'string', ['sigmoid', 'relu', 'tanh'])
print("user selected: ", activation_fn_hidd)
activation_fn_hidd = activation_fn_hidd.lower()

activation_fn_output = process_input("Please choose activation function for the output layer "+
                            "\nAvailable Activations: [sigmoid, tanh]", 'string', ['sigmoid', 'tanh'])
activation_fn_output = activation_fn_output.lower()

activation_fns = [activation_fn_hidd, activation_fn_output]

epochs = process_input("Enter number of epochs:", 'int')
learning_rate = process_input("Enter learning rate:", 'float')
# training our model
from MLP import MultilayerPerceptronNW
mlp = MultilayerPerceptronNW(4, hidden_layers, 1, activation_type=activation_fns)

mlp.train_mlp(X,y,int(epochs),float(learning_rate))
# Drawing our MLP Network

# visualize our neural network
from DrawNN import DrawNN
layers = [4] + hidden_layers + [1]
network = DrawNN(layers)
network.draw()
