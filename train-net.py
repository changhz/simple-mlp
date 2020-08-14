#!/usr/bin/env python
from sys import argv
from NeuralNetwork0 import NeuralNetwork0

import pprint
pp = pprint.PrettyPrinter(width=40)

import timeit
start = timeit.default_timer()

import argparse

parser = argparse.ArgumentParser(description='Test a neural network')
parser.add_argument('--hidden-layers', default=1)
parser.add_argument('--no-bias', nargs='?')
parser.add_argument('--silent', nargs='?')
parser.add_argument('--verbose', nargs='?')
parser.add_argument('--init-w', default=1)
parser.add_argument('--gamma', default=1)
parser.add_argument('--emax', default=0.001)
parser.add_argument('--imax', default=1000)
parser.add_argument('--attempts', default=10)
parser.add_argument('--train-set', nargs=1)
parser.add_argument('--export', nargs='?')

args = parser.parse_args()
va = vars(args)
#print(va)

import json

train_set = json.load(open(va['train_set'][0]))

input_size = len(train_set[0][0])
output_size = len(train_set[0][1])
hidden_layers = int(va['hidden_layers'])

use_bias = '--no-bias' not in argv
silent = '--silent' in argv
verbose = '--verbose' in argv
export = '--export' in argv

init_w = int(va['init_w'])
gamma = float(va['gamma'])

e_max = float(va['emax'])
i_max = int(va['imax'])

attempts = int(va['attempts'])

shape = [input_size]
for h in range(hidden_layers - 1, -1, -1):
    shape.append(h + input_size + 1)

shape.append(output_size)
info = ('Train settings: e_max: %.3f, i_max: %d, gamma: %.3f' % (e_max, i_max, gamma))
if not silent:
    print('Shape:', shape, ', Use bias: %s' % use_bias)
    print(info)

#raise

success = False
for i in range(attempts):
    net = NeuralNetwork0(shape, init_w, use_bias)

    e_trained, iterations = net.train(train_set, e_max, i_max, gamma)

    success = e_max >= e_trained
    if success:
        break

if success:
    print('Achieved in %d attempt(s) after %d iterations' % (i + 1, iterations))
    if export:
        network = {
            'shape': shape,
            'init_w': init_w,
            'use_bias': use_bias,
            'weights': net.weights,
            'bias': net.bias,
            'info': info,
        }
        with open(va['export'], 'w') as outfile:
            json.dump(network, outfile)
else:
    print('Failed to meet max error %.3f in %d loops after %d attempt(s)' % (e_max, i_max, attempts))

if verbose:
    for t in train_set:
        net.calc(t[0])
        print(net.out())
    #pp.pprint(net.bias)
    pp.pprint(net.weights[0])
    pp.pprint(net.weights[-1])

stop = timeit.default_timer()
print('Time elapsed: %.3f secs' % (stop - start))
