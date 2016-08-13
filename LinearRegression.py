#!/usr/bin/env python3
#
# A quick implementation of Linear Regression (LinearRegression.py)
# Copyright (C) 2015-2016 Stephen Makonin. All Right Reserved.

from theano import tensor as T, In, function, shared, config
import numpy as np

class linreg(object):
        """A quick implementation of Linear Regression
        """

        def __init__(self):
            self.X = T.scalar('X')
            self.Y = T.scalar('Y')
            self.W = shared(np.asarray(0., dtype=config.floatX), 'W')

            self.y = self.X * self.W
            self.cost = T.mean(T.sqr(self.y - self.Y))
            self.gradient = T.grad(cost=self.cost, wrt=self.W)
            self.update = self.W - self.gradient * 0.01

            self.train = function(inputs=[self.X, self.Y], outputs=self.cost, updates=[(self.W, self.update)], allow_input_downcast=True, name='train')

def main():
    #import matplotlib
    #matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    train_x = np.linspace(-1, 1, 101)
    train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.33

    lr = linreg()

    for i in range(100):
    	for (x, y) in zip(train_x, train_y):
    		lr.train(x, y)

    fig, ax = plt.subplots()
    ax.scatter(train_x, train_y)
    ax.plot(train_x, lr.W.get_value() * train_x, color='red')
    fig.show()

if __name__ == '__main__':
    main()
