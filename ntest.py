import json
import argparse
import timeit
from NeuralNetwork0 import NeuralNetwork0

import pprint
pp = pprint.PrettyPrinter(width=40)

start = timeit.default_timer()

parser = argparse.ArgumentParser(description='Test a neural network')
parser.add_argument('--hidden-layers', default=1)
parser.add_argument('--no-bias', nargs='?', default=False)
parser.add_argument('--init-w', default=1)
parser.add_argument('--gamma', default=1)
parser.add_argument('--emax', default=0.001)
parser.add_argument('--imax', default=1000)
parser.add_argument('--train-set', nargs=1)
parser.add_argument('--export', nargs='?')

args = parser.parse_args()
va = vars(args)
# print(va)

train_set = json.load(open(va['train_set'][0]))

input_size = len(train_set[0][0])
output_size = len(train_set[0][1])
hidden_layers = int(va['hidden_layers'])

use_bias = True if va['no_bias'] == False else False

init_w = int(va['init_w'])
gamma = float(va['gamma'])

e_max = float(va['emax'])
i_max = int(va['imax'])

shape = [input_size]
for h in range(hidden_layers - 1, -1, -1):
    shape.append(h + input_size + 1)

shape.append(output_size)
print('Shape:', shape, ', Use bias: %s' % use_bias)
print('init_w: %d, e_max: %.3f, i_max: %d, gamma: %.3f' %
      (init_w, e_max, i_max, gamma))

# raise

for i in range(10):
    net = NeuralNetwork0(shape, init_w, use_bias)

    e_trained = net.train(train_set, e_max, i_max, gamma)

    success = e_max >= len(e_trained)
    if success:
        break

if success:
    print('Achieved in %d attempt(s) after %d iterations' %
          (i + 1, net.i_last_train))
    if va['export'] != None:
        network = {
            'shape': shape,
            'init_w': init_w,
            'use_bias': use_bias,
            'weights': net.weights,
            'bias': net.bias,
        }
        with open(va['export'], 'w') as outfile:
            json.dump(network, outfile)
else:
    print('Failed')

for t in train_set:
    net.calc(t[0])
    print(net.out())

# pp.pprint(net.bias)
# pp.pprint(net.weights[0])
# pp.pprint(net.weights[-1])

stop = timeit.default_timer()
# print('\n')
print('Time elapsed: %.3f secs' % (stop - start))
