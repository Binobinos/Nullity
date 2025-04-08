import math
from math import exp
from random import gauss


def relu(x: float | int, threshold: float | int = 0.0, types: str = "norm"): 
    """Функция активации relu: f(x) = 0 if x < 0 else x [0,+inf) """
    return (x if x > threshold else 0) if types == "norm" else (1.0 if x > 0 else 0.0)


def sigmoid(x: float | int, types: str = "norm"): 
    """ Функция активации sidmoid-а, преобразует значения в диапзон [0,1] """
    return (1 / (1 + exp(-x))) if types == "norm" else (x * (1.0 - x))


def mse(predicted: list, target: list) -> float:
    """ Расчитывает ошибку между нужным вектором и выходным (обучение с учителем) """
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)

class Layer:
    """ Класс слоя, содержит нейроны и функцию активации для слоя"""
    __slots__ = ['neurons', 'act_func']
    def __init__(self, *, neurons, func): self.neurons, self.act_func = list(0 for _ in range(neurons)), func


class Neuralnetwork:
    """ Основной класс полносвязной нейросети, содержит вае функции активации и слом
    содержит функции для инициализации весов и прямого прохода"""
    __slots__ = ['layers', 'weights', 'biases', 'functions']

    def __init__(self):
        self.layers, self.weights, self.biases, self.functions = [], [], [], {"RELU": relu, "SIGMOID": sigmoid}

    def add_layer(self, neurons: int, func: str):
        """ Добавляет новый слой(первый входной ,последний выходной. остальные скрытые """
        self.layers.append(Layer(neurons=neurons, func=self.functions[func]))

    def return_weights(self):
        """ Возвращает веса нейросети """
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
        """ Инициализация весов и biases для нейроной сети"""
        self.weights, self.biases = [], []
        for i in range(len(self.layers) - 1):
            self.weights.append([[gauss(0, math.sqrt(
                2.0 / (len(self.layers[i].neurons) + len(self.layers[i + 1].neurons)))) for _ in
                                  range(len(self.layers[i + 1].neurons))] for _ in
                                 range(len(self.layers[i].neurons))])
            self.biases.append([0 for _ in range(len(self.layers[i + 1].neurons))])
    def neuron(self, input_vector):
        """ Прямой проход"""
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

