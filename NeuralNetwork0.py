from functions import sigmoid as activate
from copy import deepcopy
import pprint
import random

pp = pprint.PrettyPrinter(width=40)

class NeuralNetwork0:
    def __init__(self, shape=(4, 5, 6, 3), init_w=1, use_bias=True):
        self.shape = shape
        self.use_bias = use_bias

        layers = list()
        for n in shape:
            layers.append([0] * n)
        self.layers = layers

        bias = deepcopy(layers)

        weights = list()
        j = 1                               # layers[1] is the first hidden layer
        while j < len(layers):
            wj = list()
            for m in range(len(layers[j])):
                wm = list()
                bias[j][m] = random.uniform(-init_w, init_w) if use_bias else 0
                for n in range(len(layers[j-1])):
                    w = random.uniform(-init_w, init_w)   # initial weights matter
                    wm.append(round(w, 2))
                wj.append(wm)
            j += 1
            weights.append(wj)
        self.weights = weights
        self.bias = bias
        #pp.pprint(bias)

    def calc(self, inpt):
        layers = deepcopy(self.layers)
        layers[0] = list(inpt)

        bias = self.bias

        weights = deepcopy(self.weights)

        j = 1
        while j < len(layers):
            i = j-1
            for m in range(len(layers[j])):
                for n in range(len(layers[i])):
                    layers[j][m] += layers[i][n] * weights[i][m][n]
                b = 0
                if self.use_bias and j < len(layers):
                    b = bias[j][m]
                    #print(j, m, b)
                layers[j][m] = activate(layers[j][m] + b)
            j += 1

        del weights
        self.output = layers

    def backprop(self, target, gamma=1):
        output = self.output

        layers = deepcopy(self.layers)

        for m, o in enumerate(output[-1]):
            layers[-1][m] = (o - target[m]) * o * (1 - o)

        bias = self.bias
        weights = self.weights

        j = len(layers) - 2
        while j > 0:
            l = j+1
            for m in range(len(layers[j])):
                o = output[j][m]
                for n in range(len(layers[l])):
                    dl = layers[l][n]
                    wjl = weights[j][n][m]      # not sure
                    layers[j][m] += dl * wjl * o * (1 - o)
                    #print('j:%d, m:%d, n:%d => %.3f' % (j, m, n, wjl))
            j -= 1

        j = 1
        while j < len(layers):
            i = j-1
            for m in range(len(layers[j])):
                d = layers[j][m]
                for n in range(len(output[i])):
                    o = output[i][n]
                    weights[i][m][n] += -1 * gamma * o * d
                    if self.use_bias:
                        bias[j][m] += -1 * gamma * d
                        #print(bias[j][m])
                    #print('j:%d, m:%d, n:%d => %.3f' % (j, m, n, wjl))
            j += 1

    def train(self, train_set, e_max=0.001, i_max=1000, gamma=1):
        inputs = [ train_set[p][0] for p in range(len(train_set)) ]
        targets = [ train_set[p][1] for p in range(len(train_set)) ]
        error = 0
        for i in range(i_max + 1):
            outputs = list()
            error = 0
            for l in range(len(inputs)):
                self.calc(inputs[l])
                self.backprop(targets[l], gamma)
                outputs.append(list(self.output[-1]))

            for l in range(len(outputs)):
                for n in range(len(outputs[l])):
                    error += ((targets[l][n] - outputs[l][n]) ** 2) / 2

            error /= len(outputs) * len(outputs[0])

            if error <= e_max:
                break
        #print('i:%d, e:%.3f' % (i, error))
        return error, i

    def out(self):
        o = self.output[-1]
        return [ round(n, 3) for n in o ]

if __name__ == '__main__':
    net = NeuralNetwork0()
    net.calc((0, 0, 0, 1))
    net.backprop((1, 0, 0))

