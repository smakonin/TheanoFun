#!/usr/bin/env python3
#
# An implementation of the original knobby ADALINE experiment (KnobbyAdaline.py)
# Copyright (C) 2015-2016 Stephen Makonin. All Right Reserved.
#
# See Widrow's YouTub videos:
#    https://www.youtube.com/watch?v=hc2Zj55j1zU
#    https://www.youtube.com/watch?v=skfNlwEbqck

from theano import tensor as T, pp, In, function, shared, config
import numpy as np

class adaline(object):
    """ADALINE (Adaptive Linear Neuron or later Adaptive Linear Element)

    A 1950s single-layer artificial neural network and the name of the physical
    device that implemented this network. The network Is trained using LMS. It
    was developed by Professor Bernard Widrow and his graduate student Ted Hoff
    at Stanford University.

    See: https://en.wikipedia.org/wiki/ADALINE (not the best article)
    """

    def __init__(self, N):
        self.N = N  # the number of inputs/weights
        self.X = T.vector('X') # the input vector
        self.d = T.scalar('d') # the desired output
        self.μ = T.dscalar('μ') # the learning rate

        self.μ = shared(np.asarray(0.005, dtype=config.floatX), 'μ')
        self.W = shared(np.asarray([0.] * N, dtype=config.floatX), 'W')

        self.y = T.sum(self.X * self.W) # the response
        self.q = T.sgn(self.y) # the output
        self.e = self.d - self.y # the error
        self.lms = self.W + self.X * 2 * self.μ * self.e # the least means squred algorithm

        self.activate = function([self.X], self.y, name='activate')
        self.output = function(inputs=[self.X], outputs=self.q, name='output')
        self.train = function(inputs=[self.X, self.d], outputs=self.y, updates=[(self.W, self.lms)], allow_input_downcast=True, name='train')

def main():
    print()
    print()
    print('A recreation of the original')
    print('Knobby ADALINE using the LMS algorithm')
    print('======================================')

    print()
    print('Knobby Training using 4×4 T and J patterns:')

    print()
    print('\t●●●○ ○○●○ ○●●● ○○○● ○○○○ ○○○○ ●○○○ ●●●●')
    print('\t○●○○ ○○●○ ○○●○ ○○○● ●○○○ ●●●● ●●●● ○○○●')
    print('\t○●○○ ●○●○ ○○●○ ○●○● ●●●● ○○○● ●○○○ ○○●●')
    print('\t○●○○ ●●●○ ○○●○ ○●●● ●○○○ ○○●● ○○○○ ○○○○ ')
    print()
    print('\tWhere ● = 1 and ○ = -1')
    print()

    # columns are: name, input pattern vector, desired response, trained?
    t = [['T1', [ 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1],  1, False],
         ['J1', [-1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1], -1, False],
         ['T2', [-1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1],  1, False],
         ['J2', [-1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1], -1, False],
         ['T3', [-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1],  1, False],
         ['J3', [-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1], -1, False],
         ['T4', [ 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1],  1, False],
         ['J4', [ 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1], -1, False]]

    training_round = 0
    minimal_response = 0.5

    knobby = adaline(len(t[0][1]))

    while(len(t) != sum(list(zip(*t))[3])):
        for i in range(len(t)):
            input_pattern = t[i][1]
            desired_response = t[i][2]
            is_trained = t[i][3]

            if not is_trained:
                knobby.train(input_pattern, desired_response)

        for i in range(len(t)):
            t[i][3] = not -minimal_response < knobby.activate(t[i][1]) < minimal_response

        training_round += 1
        print('\tTraining Round %3d: Correct pattern responses %2d, %5.1f%% complete.' % (training_round, sum(list(zip(*t))[3]), sum(list(zip(*t))[3]) / len(t) * 100))

    print()
    print('Knobby Results:')
    print()

    for tt in t:
        print('\tPattern %s has reponse %c with an activate of  %19.16f' % (tt[0], '●' if knobby.output(tt[1]) == 1 else '○', knobby.activate(tt[1])))

    print()
    print()

if __name__ == '__main__':
    main()
