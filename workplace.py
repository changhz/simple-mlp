from NeuralNetwork0 import NeuralNetwork0
#from pprint import pprint

import json

data = json.load(open('./data/4x4/net.json'))

net = NeuralNetwork0(data['shape'], data['init_w'], data['use_bias'])
net.weights = data['weights']
net.bias = data['bias']
net.calc([
    1,1,
    1,1,
])
print(net.out())
