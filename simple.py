"""
    A simple supervised neural network
    which distinguishes patterns: diagonal, vertical and horizontal
"""
import numpy as np
import pprint

pp = pprint.PrettyPrinter(width=40)

structure = (4, 5, 3)
#structure = (25, 30, 3)

w_layers = list()
d_layers = list()
for i, n in enumerate(structure):
    if i > 0:
        d_layers.append([0] * n)
    if i+1 < len(structure):
        next_n = structure[i+1]
        w_layers.append(np.random.random((n, next_n)))
#pp.pprint(w_layers)
#print(d_layers)

#assert False

def sigmoid(x):
    return 1 / (1 + 1 / np.power(np.e, x))
#print(sigmoid(0))

outputs = list()

# Calculation
def calc(inpt):
    inpt_ = inpt.copy()
    for l, weights in enumerate(w_layers):
        outpt = [0] * len(weights[l+1])
        for i, r in enumerate(weights):
            for j, w in enumerate(r):
                #print('l:%d, i:%d, j:%d => %.3f' % (l, i, j, w))
                outpt[j] += inpt_[i] * w
                #print(outpt[j])
                #print('\n')
        #print(outpt)
        inpt_ = [ sigmoid(x) for x in outpt ]
        outputs.append(inpt_)
    return inpt_

"""
input_list = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]
"""
input_list = [
    [0, 1],
    [1, 0],
]

inpt = list()
for r in input_list:
    for x in r:
        inpt.append(x)
print(inpt)
outputs.append(inpt)

outpt = calc(inpt)
print(outpt)
#pp.pprint(outputs)
print('\n')

# Learning
l_rate = 0.9
target = [1, 0, 0]

def calc_delta(o, t):
    return (o - t) * o * (1 - o)

d_layers[len(w_layers) - 1] = [ calc_delta(outpt[i], t) for i, t in enumerate(target) ]
#print(d_layers)

for l in range(len(w_layers)-1, -1, -1):
    weights = w_layers[l]
    #print(update)
    for i in range(len(weights)-1, -1, -1):
        r = weights[i]
        for j in range(len(r)-1, -1, -1):
            #print(d_layers[l][j])
            w = r[j]
            #print('l:%d, i:%d, j:%d => %.3f' % (l, i, j, w))
            #print(outputs[l][i])
            w_layers[l][i, j] += -1 * l_rate * outputs[l][i] * d_layers[l][j]
            w = w_layers[l][i, j]
            #print('l:%d, i:%d, j:%d => %.3f' % (l, i, j, w))
            #print('\n')
            if l > 0:
                d_layers[l-1][i] += d_layers[l][j] * w * outpt[j] * (1 - outpt[j])
                #print(d_layers[l-1][i])
#pp.pprint(w_layers)
print(calc(inpt))

