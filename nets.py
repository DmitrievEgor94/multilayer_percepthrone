import numpy as np
from scipy.special import softmax
from scipy.stats import logistic
from collections.abc import Sequence

sigmoid = logistic.cdf


def sigmoid_derivative(x):
    y = sigmoid(x)
    return (1 - y) * y


class MLPClassification:
    def __init__(self, network_shape: Sequence[int]):
        self.layers = []

        for i in range(len(network_shape)):
            if i == 0:
                self.layers.append(np.zeros(network_shape[i] + 1))
            else:
                self.layers.append(np.zeros(network_shape[i]))

        # массив весов
        self.weights = []

        for i in range(len(network_shape) - 1):
            # сделаем инициализацию Ксавьера https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.xavier_uniform_
            bound = (6 / (self.layers[i].size + self.layers[i + 1].size)) ** (1 / 2)
            # добавляем 1 для весов смещения
            self.weights.append(np.random.uniform(-bound, bound, (self.layers[i].size, self.layers[i + 1].size)))

        # для метода моментов сохраняем предыдущие производные
        self.dw = [0, ] * len(self.weights)

    def forward(self, x: np.array):
        # инициализация контекстного слоя
        self.layers[0] = np.hstack((x, 1))

        for i in range(1, len(self.weights)+1):
            self.layers[i][...] = sigmoid(np.dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[-1], softmax(self.layers[-1])

    def backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []

        # подсчет ошибок на выходном слое (используется производная от кросс-энропии с софтмаксом)
        error = target - self.layers[-1]
        delta = error * sigmoid_derivative(self.layers[-1])
        deltas.append(delta)

        # Ошибка на скрытых слоях
        for i in range(1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * sigmoid_derivative(self.layers[i])
            deltas.insert(0, delta)

        # обновляем веса
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            # print(dw)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

