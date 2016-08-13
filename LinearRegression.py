#!/usr/bin/env python3
#
# A quick implementation of Linear Regression (LinearRegression.py)
# Copyright (C) 2015 Stephen Makonin. All Right Reserved.

from theano import tensor as T, In, function, shared, config
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1, 1, 101)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.33

X = T.scalar('X')
Y = T.scalar('Y')
W = shared(np.asarray(0., dtype=config.floatX), 'W')

y = X * W
cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=W)
updates = [[W, W - gradient * 0.01]]

train = function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True, name='train')

for i in range(100):
	for (x, y) in zip(train_x, train_y):
		train(x, y)

fig, ax = plt.subplots()
#fit = np.polyfit(train_x, train_y, deg=1)
#ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(train_x, train_y)
ax.plot(train_x, W.get_value() * train_x, color='red')
fig.show()
