#!/usr/bin/env python3
#
# An implementation of the original knobby ADALINE experiment (KnobbyAdaline.py)
# Copyright (C) 2015 Stephen Makonin. All Right Reserved.
#
# See Widrow's YouTub videos:
#    https://www.youtube.com/watch?v=hc2Zj55j1zU
#    https://www.youtube.com/watch?v=skfNlwEbqck
#
# ADALINE (Adaptive Linear Neuron or later Adaptive Linear Element) is an 1950s
# single-layer artificial neural network and the name of the physical device that
# implemented this network. The network Is trained using LMS. It was developed
# by Professor Bernard Widrow and his graduate student Ted Hoff at Stanford
# University in 1960.
#
# See: https://en.wikipedia.org/wiki/ADALINE (not the best article)
#

from theano import tensor as T, pp, In, function, shared, config
import numpy as np

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

N = len(t[0][1])  # the number of inputs/weights
X = T.vector('X') # the input vector
d = T.scalar('d') # the desired output
μ = T.dscalar('μ') # the learning rate

W = shared(np.asarray([0.] * N, dtype=config.floatX), 'W')

y = T.sum(X * W) # the response
q = T.sgn(y) # the output
e = d - y # the error
lms = [[W, W + X * 2 * μ * e]] # the least means squred algorithm

activate = function([X], y, name='activate')
output = function(inputs=[X], outputs=q, name='output')
train = function(inputs=[X, d, In(μ, value=0.005)], outputs=y, updates=lms, allow_input_downcast=True, name='train')

training_round = 0
minimal_response = 0.5

while(len(t) != sum(list(zip(*t))[3])):
    for i in range(len(t)):
        input_pattern = t[i][1]
        desired_response = t[i][2]
        is_trained = t[i][3]

        if not is_trained:
            train(input_pattern, desired_response)

    for i in range(len(t)):
        t[i][3] = not -minimal_response < activate(t[i][1]) < minimal_response

    training_round += 1
    print('\tTraining Round %3d: Correct pattern responses %2d, %5.1f%% complete.' % (training_round, sum(list(zip(*t))[3]), sum(list(zip(*t))[3]) / len(t) * 100))

print()
print('Knobby Results:')
print()

for tt in t:
    print('\tPattern %s has reponse %c with an activate of  %19.16f' % (tt[0], '●' if output(tt[1]) == 1 else '○', activate(tt[1])))

print()
print()
