import numpy as np
import pprint

pp = pprint.PrettyPrinter(width=40)

def sigmoid(x):
    return 1 / (1 + 1 / np.e ** x)

def calc_delta(o, t):
    res = (o - t) * o * (1 - o)
    #print(res)
    return res

def calc_err(t, y):
    return ((t - y) ** 2) / 2

class SimpleNet:
    def __init__(self, structure=(4, 5, 3), lr=0.01):
        self.structure = structure
        self.l_len = len(structure) - 1

        self.ol = list()
        self.wl = list()
        self.dl = list()

        self.lr = lr

        for l, n in enumerate(structure):
            self.ol.append([0.0] * n)
            if l > 0:
                self.dl.append(np.array([0.0] * n))
            if l+1 < len(structure):
                next_n = structure[l+1]
                self.wl.append(np.random.random((n, next_n)))
                #self.wl.append(np.zeros((n, next_n)))
                #self.wl.append(np.array([[0.5] * next_n] * n))
        #pp.pprint(self.ol)
        #pp.pprint(self.dl)

    def calc(self, inpt):
        self.ol[0] = inpt
        for k in range(len(self.structure) - 1):
            j = k+1
            for p in range(self.structure[j]):
                #print('k:%d, p:%d => %.3f' % (k, p, 0))
                #for n, s in enumerate(self.ol[k]):
                    #print(self.wl[k][n, p])
                self.ol[j][p] = sigmoid(sum([ s * self.wl[k][n, p] for n, s in enumerate(self.ol[k]) ]))

    def train(self, inpt, target, e_max=0.01, i_max=2000):
        i = 0
        while True:
            self.calc(inpt)
            outpt = self.ol[-1]
            error = 0

            for n in range(len(outpt)):
                error += calc_err(target[n], outpt[n])

            error /= len(outpt)

            i += 1
            if error < e_max or i > i_max:
                #print('i:%d, e:%.3f' % (i, error))
                break
            self.diff(target)

    def diff(self, target):
        outpt = self.ol[-1]
        for n in range(len(target)):
            self.dl[-1][n] = calc_delta(outpt[n], target[n])

        for j in range(self.l_len - 2, -1, -1):
            l = j+1
            for p, d in enumerate(self.dl[j]):
                o = self.ol[j+1][p]
                for q, d_ in enumerate(self.dl[l]):
                    w = self.wl[l][p, q]
                    self.dl[j][p] += w * d_ * o * (1 - o)
                    #print('j:%d, p:%d => %.3f' % (j, p, d))

        for i in range(len(self.structure) - 1):
            j = i+1
            for p in range(self.structure[i]):
                for q in range(self.structure[j]):
                    self.wl[i][p, q] += -1 * self.lr * self.ol[i][p] * self.dl[i][q]
                    #print('i:%d, p:%d, q:%d' % (i, p, q))
        #pp.pprint(self.wl)

if __name__ == '__main__':
    import timeit
    #import json

    start = timeit.default_timer()

    net = SimpleNet(structure=(4, 8, 2))
    #pp.pprint(net.wl)

    net.train((0, 1, 1, 0), [1,0])
    net.train((1, 0, 0, 1), [0,1])
    net.train((0, 0, 0, 0), [0,0])
    net.train((1, 1, 1, 1), [0,0])
    net.train((1, 0, 1, 0), [1,1])
    net.train((0, 1, 0, 1), [1,1])
    net.train((1, 1, 0, 0), [1,0])
    net.train((0, 0, 1, 1), [0,1])

    net.calc((0, 0, 0, 0))
    pp.pprint(net.ol[-1])
    #pp.pprint(net.wl)

    """
    weights = [ w.tolist() for w in net.wl ]
    with open('weights.json', 'w') as outfile:
        json.dump(weights, outfile)
    """

    stop = timeit.default_timer()
    print('\n')
    print('%.3f secs' % (stop - start))

