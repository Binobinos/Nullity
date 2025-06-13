import logging
import math
from math import exp
from random import gauss
from typing import List, Union, Tuple

from tqdm import tqdm


def relu(x: float | int, threshold: float | int = 0.0, types: str = "norm"):
    """
    Функция активации relu: f(x) = 0 if x < 0 else x [0,+inf)
    :param x: параметр x
    :param threshold: порог активации функции
    :param types: Тип функции. norm для обычного режима и derivative для полихводной
    :return: Знацение функции RELU для параметра x
    """
    return (x if x > threshold else 0) if types == "norm" else (1.0 if x > threshold else 0.0)


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
        """ Добавляет новый слой (первый входной, последний выходной. Остальные скрытые) """
        self.layers.append(Layer(neurons=neurons, func=self.functions[func]))

    def return_weights(self):
        """ Возвращает веса нейросети """
        return [self.weights, self.biases]

    def update_weights(self, inputs, targets, learning_rate=0.01) -> None:
        """
        Проход Градиентного спуска
        :param inputs: Входные данные
        :param targets: Выходной слой
        :param learning_rate: скорость обучения
        :return: None
        """
        outputs = self.neuron(inputs)
        errors = [t - o for t, o in zip(targets, outputs[-1])]

        for i in reversed(range(len(self.layers) - 1)):
            grad = [e * self.layers[i + 1].act_func(outputs[i + 1][j], types="derivative")
                    for j, e in enumerate(errors)]

            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    delta = learning_rate * grad[k] * outputs[i][j]
                    self.weights[i][j][k] += delta

            new_errors = [0.0] * len(self.layers[i].neurons)
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    new_errors[j] += errors[k] * self.weights[i][j][k]
            errors = new_errors

    def calculating_weights(self):
        """Инициализация весов и biases для нейронной сети."""
        self.weights, self.biases = [], []
        for i in range(len(self.layers) - 1):
            # Xavier-инициализация для сигмоиды, He для ReLU
            if self.layers[i + 1].act_func.__name__ == "sigmoid":
                limit = math.sqrt(6.0 / (len(self.layers[i].neurons) + len(self.layers[i + 1].neurons)))
            else:  # ReLU
                limit = math.sqrt(2.0 / len(self.layers[i].neurons))

            self.weights.append([
                [gauss(0, limit) for _ in range(len(self.layers[i + 1].neurons))]
                for _ in range(len(self.layers[i].neurons))
            ])
            self.biases.append([0 for _ in range(len(self.layers[i + 1].neurons))])

    def neuron(self, input_vector):
        """ Прямой проход"""
        outputs = [input_vector]
        for l_id in range(len(self.layers) - 1):
            current_layer_output = []
            current_weights = self.weights[l_id]  # Weights for this layer
            current_biases = self.biases[l_id]  # Biases for this layer

            # Check dimensions
            if len(current_weights) != len(outputs[-1]):
                raise ValueError(
                    f"Weight matrix dimension mismatch. Expected {len(outputs[-1])} input connections, got {len(current_weights)}")

            for neu_id in range(len(self.layers[l_id + 1].neurons)):
                # Calculate weighted sum
                weighted_sum = 0
                for inp_id in range(len(outputs[-1])):
                    try:
                        weighted_sum += outputs[-1][inp_id] * current_weights[inp_id][neu_id]
                    except IndexError:
                        raise IndexError(f"Weight index out of range. Layer {l_id}, input {inp_id}, neuron {neu_id}")

                weighted_sum += current_biases[neu_id]
                activated = self.layers[l_id + 1].act_func(weighted_sum)
                current_layer_output.append(activated)
            outputs.append(current_layer_output)
        return outputs

    def train(self, data: List[Tuple[list, list]],
              epoch: int,
              learning_rate: Union[int, float]) -> None:
        """
        Обучение нейросети на входных данных
        :param data: Данные в формате списка, где каждый элемент это картеж с входным слоем и выходным
        :param epoch: Количество эпох
        :param learning_rate: скорость обучения
        :return: None
        """
        for epochs in tqdm(range(epoch)):
            for input_vector, target_vector in tqdm(data):
                try:
                    self.update_weights(input_vector, target_vector, learning_rate)
                except Exception as e:
                    print(e)

    def predict(self, data: list):
        """Предсказывает выходной слой по входным данным"""
        return self.neuron(data)[-1]
