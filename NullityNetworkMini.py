from math import exp
from random import gauss


def relu(x: float | int, threshold: float | int = 0.0, types: str = "norm"):
    return (x if x > threshold else 0) if types == "norm" else (1.0 if x > 0 else 0.0)


def sigmoid(x: float | int, types: str = "norm"):
    return (1 / (1 + exp(-x))) if types == "norm" else (x * (1.0 - x))


def mse(predicted: list, target: list) -> float:
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)


class Layer:
    __slots__ = ['neurons', 'act_func']

    def __init__(self, *, neurons, func): self.neurons, self.act_func = list(0 for _ in range(neurons)), func


class Neuralnetwork:
    __slots__ = ['layers', 'weights', 'biases', 'functions']

    def __init__(self):
        self.layers, self.weights, self.biases, self.functions = [], [], [], {"RELU": relu, "SIGMOID": sigmoid}

    def add_layer(self, neurons: int, func: str):
        self.layers.append(Layer(neurons=neurons, func=self.functions[func]))

    def return_weights(self):
        return self.weights

    def update_weights(self, inputs, targets, learning_rate=0.01):
        outputs = self.neuron(inputs)
        errors = [t - o for t, o in zip(targets, outputs[-1])]
        for i in reversed(range(len(self.layers) - 1)):
            errors = [e * self.layers[i + 1].act_func(o, types="derivative") for e, o in zip(errors, outputs[i + 1])]
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += learning_rate * errors[k] * outputs[i][j]

    def calculating_weights(self):
        self.weights, self.biases = [], []
        for i in range(len(self.layers) - 1):
            self.weights.append([[gauss(0, 1) for _ in range(len(self.layers[i + 1].neurons))] for _ in
                                 range(len(self.layers[i].neurons))])
            self.biases.append([0.01 for _ in range(len(self.layers[i + 1].neurons))])

    def neuron(self, input_vector):
        outputs = [input_vector]
        for l_id in range(len(self.layers) - 1):
            current_layer_output = []
            for neu_id in range(len(self.layers[l_id + 1].neurons)):
                weighted_sum = sum(
                    outputs[-1][inp_id] * self.weights[l_id][inp_id][neu_id] for inp_id in range(len(outputs[-1]))) + \
                               self.biases[l_id][neu_id]
                activated = self.layers[l_id + 1].act_func(weighted_sum)
                current_layer_output.append(activated)
            outputs.append(current_layer_output)
        return outputs
