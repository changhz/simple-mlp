from NeuralNetwork0 import NeuralNetwork0

import timeit
import pprint

pp = pprint.PrettyPrinter(width=40)

start = timeit.default_timer()

def test0():
    input_set = (
        (
            0,0,1,1,0,
            0,1,0,1,0,
            0,1,0,1,0,
            0,1,0,1,0,
            0,1,1,0,0,
        ),
        (
            0,1,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
        ),
        (
            0,1,1,1,0,
            0,1,0,1,0,
            0,0,1,1,0,
            0,1,0,0,0,
            0,1,1,1,0,
        ),
        (
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0,
            0,1,0,0,0,
            0,1,1,1,0,
        ),
        (
            0,1,1,1,0,
            0,0,0,1,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,1,1,1,0,
        ),
        (
            0,1,1,1,0,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
        ),
        (
            0,1,1,1,0,
            0,1,0,0,0,
            0,1,1,1,0,
            0,0,0,1,0,
            0,1,1,1,0,
        ),
        (
            0,1,1,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
        ),
        (
            0,1,1,1,0,
            0,1,0,1,0,
            0,1,0,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
        ),
        (
            0,0,1,0,0,
            0,1,1,0,0,
            1,1,1,1,0,
            0,0,1,0,0,
            0,0,1,0,0,
        ),
        (
            0,1,1,0,0,
            0,1,1,0,0,
            0,1,1,1,0,
            0,0,1,0,0,
            0,0,1,0,0,
        ),
        (
            0,1,1,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
            0,0,0,1,0,
            0,0,0,1,0,
        ),
    )
    target_set = (
        (0,0,0,0),
        (0,0,0,1),
        (0,0,1,0),
        (0,0,1,0),
        (0,0,1,1),
        (0,1,1,1),
        (0,1,0,1),
        (1,0,0,0),
        (0,0,0,0),
        (0,1,0,0),
        (0,1,0,0),
        (1,0,0,1),
    )
    #print(len(input_set))
    #print(len(target_set))
    net = NeuralNetwork0((25, 30, 4), 1)
    net.train(input_set, target_set, e_max=0.001, i_max=1000, gamma=1)

    """
    for i in input_set:
        net.calc(i)
        print(net.out())
    """

    net.calc((
        0,0,1,0,0,
        0,1,1,0,0,
        0,1,1,1,0,
        0,0,1,0,0,
        0,0,1,0,0,
    ))
    print(net.out())

    pp.pprint(net.weights[0])
    pp.pprint(net.weights[-1])

test0()

stop = timeit.default_timer()
#print('\n')
print('Time elapsed: %.3f secs' % (stop - start))
