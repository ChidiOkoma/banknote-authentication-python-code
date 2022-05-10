import time

import numpy as np
import activation_fn as act


class MultilayerPerceptronNW:
    def __init__(self, input_layer_len=4, hidden_layer_len=[10], output_layer_len=1, weight_min=-0.5, wight_max=0.5, activation_type = ['relu', 'sigmoid']):
        # number of inputs (features) to the network,
        # this will also determine number of neurons in input layer
        self.input_layer_len = input_layer_len

        # number of hidden layers with number of neurons
        # this is expected as a list [],
        # e.g. [12,8] will mean two hidden layers wit 12 neurons in first hidden layer
        # and 8 neurons in second hidden layer
        self.hidden_layer_len = hidden_layer_len

        # this is number of outputs, for our mlp its 1
        self.output_layer_len = output_layer_len

        # This will calculate layer size which len of all layers summed
        layers = [self.input_layer_len] + self.hidden_layer_len + [self.output_layer_len]

        # calculating our nn complexity which we did size of hidden layer + 2(one input layer + one output layer)
        # we will use this to get output layer when applying activations
        self.nn_size = len(self.hidden_layer_len) + 2

        #print("layers: ", layers)

        # initiating weight ranges to configured
        self.weight_min = weight_min
        self.weight_max = wight_max

        # initiate weights and biases
        weights, biases = self.initiate_weights(layers)
        self.weights = weights
        self.biases = biases

        # initiate derivatives nd array
        weight_derivatives, bias_derivatives = self.calculate_derivatives(layers)
        self.weight_derivatives = weight_derivatives
        self.bias_derivatives = bias_derivatives

        # initiate activations nd array
        self.activations = self.calculate_activations(layers)

        # initiating activation function typeof our nn
        self.activation_type = activation_type

    """
        This Function will initiate weights and biases to min/max configured
        it takes layers as input
    """
    def initiate_weights(self, layers):
        # initiate weights
        weights = []
        biases = []
        for layer in range(len(layers) - 1):
            w = np.random.uniform(low=self.weight_min, high=self.weight_max,
                                  size=(layers[layer], layers[layer + 1]))
            b = np.zeros((layers[layer + 1], 1))

            print("w{} shape is {}".format(layer+1, w.shape))
            print("b{} shape is {}".format(layer+1, b.shape))

            # print("W{} shape is {}".format(layer, w.shape))
            # print("W{} range is between {} and {}".format(layer, w.min(), w.max()))
            # print("b{} shape is {}".format(layer, b.shape))
            # print("weights: ", w)
            weights.append(w)
            biases.append(b)
        # print("initialized weights: ", weights)
        # print("initialized biases: ", biases)
        return weights, biases

    """
        This method initiates derivitives as same size of the weights and biases
        It will initialize it to 0
        it will return tuple of numpy arrays (weight_derivatives, bias_derivatives) 
    """
    def calculate_derivatives(self, layers):
        # initiating dZ
        weight_derivatives = []
        bias_derivatives = []
        # for i in range(len(layers) - 1):
        #    d = np.zeros((layers[i], layers[i + 1]))
        # print("**weights**: ", self.weights)
        # print("derivitives: ",d)

        #    derivatives.append(dw)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dw = np.zeros_like(w)
            db = np.zeros_like(b)

            print("dw{} shape is {}".format(i+1, dw.shape))
            print("db{} shape is {}".format(i+1, db.shape))

            weight_derivatives.append(dw)
            bias_derivatives.append(db)
        # print("initialized derivative weights: ", weight_derivatives)
        # print("initialized derivative biases: ", bias_derivatives)
        return weight_derivatives, bias_derivatives

    """
        This method initiates activations
        
    """
    def calculate_activations(self, layers):
        # initiating Activations A
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            # print("A{} shape is {}".format(i+1, a.shape))
            activations.append(a)
        # print("activations: ", activations)
        return activations

    """
        This method applies performs forward pass to Neural Network
        it takes input features and return list of activations
    """
    def feed_forward(self, inputs):
        # for input layer, inputs to the network activates the our mlp
        # initializing it to input (features)
        self.activations[0] = inputs
        A = inputs

        # setting activation type to instance so that
        # derivative is applied for the one used in forward_propogate
        activation_fn = self.activation_type[0]
        # calculating the activations A (A1,A2,...An) for all weights and storing in nd array
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate net inputs (z) for given layer
            #print("Activations shape: ", A.shape)
            #print("Weight Transposed shape", w.T.shape)

            # calculating Z
            Z = np.matmul(w.T, A) + b
            # print("Net input (z) shape for layer {} is {}".format(i, z.shape))
            if i == (self.nn_size -2):
                activation_fn = self.activation_type[1]
               # print("Applying activation fun {} for A{}".format(activation_fn, i+1))
            #else:
               # print("Applying activation fun {} for A{}".format(activation_fn, i+1))

            # calculate the activations A
            A = self.get_activation(activation_fn, Z)
            #print("A{} is {}".format(i+1,A))

            # storing Activations A in instance for later use
            # we are associating A with W so A0 has the input features, we are storing A1 as the first activations
            self.activations[i + 1] = A

        # return the output (last Activation A)
        return A

    """
        This method performs back propogation on our network
    """
    def back_propogate(self, dz, targets, verbose=False):
        # activation_fn = self.activation_type[0]

        for i in reversed(range(len(self.weight_derivatives))):
            # output activation is i+1, getting A[i] which is the input to the output layer
            #print("length of derivitives: ", (self.weight_derivatives))

            activation_fn = self.activation_type[0]
            activations = self.activations[i]
            #print("activations = {}".format(activations))

            #print("calculating dw{} and db{}".format(i+1, i+1))
            # calculating change in weight (dw) using A[i] input to the last layer
            dw = np.matmul(activations, dz.T) / targets.shape[0]

            # calculating change in bias by sumping up dz
            db = np.sum(dz, axis=0, keepdims=True) / targets.shape[0]

            # storing my derivatives in the instance to be used later in weight updating
            self.weight_derivatives[i] = dw
            self.bias_derivatives[i] = db

            # getting derivative of the activation to be used in dz calculation
            # checkig if output layer, we are using nn_size -2 because activations fun is applied from hidden layer onwards
            print("Applying derivative of activation fun {} for A{}".format(activation_fn, i+1))

            activation_deriv = self.get_activation_deriv(activation_fn, activations)
            #print("dA{} is {}".format(i+1, activation_deriv))

            dz = np.multiply(np.matmul(self.weights[i], dz), activation_deriv)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.weight_derivatives[i]))

    """
        This method will update our weights it takes learning rate as input
    """
    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            #print("w{} is {}".format(i, w))
            #print("b{} is {}".format(i, b))
            dw = self.weight_derivatives[i]
            db = self.bias_derivatives[i]
            #print("dw{} is {}".format(i, dw))
            #print("db{} is {}".format(i, db))

            # update weights and biases
            w = w - learning_rate * dw
            b = b - learning_rate * db
            self.weights[i] = w
            self.biases[i] = b
            # print("updating wights ......")
            # print("w{} is {}".format(i, w))
            # print("b{} is {}".format(i, b))

    """
        Calculates activation based on activation function configured for our MLP
    """
    def get_activation(self, actType, z):
        if actType == 'sigmoid':
            activations = act.sigmoid(z)
        elif actType == 'relu':
            activations = act.relu(z)
        elif actType == 'tanh':
            activations = act.tanh(z)

        return activations
    
    """
        Calculates the derivative our configured activation function
    """
    def get_activation_deriv(self, actType, A):
        # get derivative of activation function
        if actType == 'sigmoid':
            activation_deriv = act.sigmoid_deriv(A)
        elif actType == 'relu':
            activation_deriv = act.relu_deriv(A)
        elif actType == 'tanh':
            activation_deriv = act.tanh_derv(A)

        return activation_deriv

    def cross_entropy_loss(self, pred, actual, eps=1e-15):

        loss = -1 * np.sum(actual * np.log(pred + eps) +
                           ((1 - actual) * np.log(1 - pred + eps)))
        loss /= actual.shape[0]
        return loss

    def mse(self, pred, actual):
        return np.average((actual - pred) ** 2)


    """
        This method will train our network.
        It will perform below below steps:
        for each epoch do:
            calculate output using forward_pass(input)
            calculate lost and print report
            perform back_propogate(output)
            update weights using learning rates
    """
    def train_mlp(self, inputs, targets, epochs, learning_rate):
        t0 = time.time()
        final_acc = 0

        print("#" * 100)
        history_loss = []
        history_acc = []
        for epoch in range(epochs):
            loss = 0
            corr = 0
            y_hat = []

            # batch learning starts here
            outputs = self.feed_forward(inputs)

            loss = self.cross_entropy_loss(outputs, targets)
            print("\Loss for epoch {} is {}".format(epoch, loss))
            
            history_loss.append(loss)
            # running back propagation
            # calculating error
            error = outputs - targets
            self.back_propogate(error, targets)

            # updating weights
            self.update_weights(learning_rate)


            # calculating accuracy
            for index,m in enumerate(outputs[0]):
                if m < 0.5:
                    y_hat.append(0)
                elif m >= 0.5:
                    y_hat.append(1)
            corr = np.sum(targets == y_hat)
            print('No. of correct predictions: ', corr)
            acc = corr / targets.shape[0]
            final_acc += corr

            history_acc.append(acc)
            print("Accuracy for epoch {} = {:.2f}%".format(epoch, acc *100))
            # batch learning ends here
        print('='*100)
        print('Time elapsed for one iteration over the whole dataset ', time.time()-t0)
        print("Number of hidden layers is {}".format(len(self.hidden_layer_len)))
        
        for i in range(len(self.hidden_layer_len)):
            print('Number of neurons in hidden layer {}, is {}'.format(i+1, self.hidden_layer_len[i]))

        print("Activation function for hidden layers: {} and for output layer: {}".format(self.activation_type[0], self.activation_type[1]))
        print('Final cost for epoch {} = {:.2f}%'.format(epoch,loss))
        print('Final Accuracy for epoch {} = {:.2f}%'.format(epoch, (final_acc / (epochs * targets.shape[0])) * 100))

        drawGraphs(epochs, learning_rate, history_loss, history_acc)

def drawGraphs (epochs, lr, loss, accuracy):
    import matplotlib.pyplot as plt

    # drawing epoches vs loss
    plt.plot(range(1,epochs+1), loss, 'r', label='loss')
    plt.title('loss when learning rate is {}'.format(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # drawing epoches vs accuracy 
    plt.plot(range(1,epochs+1), accuracy, 'g', label='accuracy')
    plt.title('accuracy when learning rate is {}'.format(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


