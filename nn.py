import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        input_layer_neurons = self.layer_sizes[0]
        hidden_layer_neurons = self.layer_sizes[1]
        output_neurons = self.layer_sizes[2]

        self.W_1 = np.random.randn(input_layer_neurons * hidden_layer_neurons).reshape(hidden_layer_neurons,
                                                                                       input_layer_neurons)  # (50, 10)
        self.b_1 = np.zeros((hidden_layer_neurons, 1))  # (50, 1)

        self.W_2 = np.random.randn(output_neurons * hidden_layer_neurons).reshape(output_neurons,
                                                                                  hidden_layer_neurons)  # (2, 50)
        self.b_2 = np.zeros((output_neurons, 1))  # (2, 1)

        # pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        # print(f"x : {x}")

        x = np.array(x)
        x = x.reshape(self.layer_sizes[0], 1)
        z1 = self.W_1 @ x + self.b_1
        # z1_hat = (z1 - np.mean(z1)) / np.std(z1)
        A1 = self.activation(z1)  # (50, 1)

        out = self.activation(self.W_2 @ A1 + self.b_2)  # (2, 1)
        # print(f"a2 shape {out.shape}")

        return out