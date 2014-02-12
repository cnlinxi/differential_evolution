import numpy
from de import DifferentialEvolution as DE
from time import sleep

'''
This file contains 12 of the 20 test functions given in Appendix A of
Storn and Price's "Differential Evolution", implemented using the
TestDifferentialEvolution class.
Functions 1-5 are unimodal.
Functions 6-10 are multimodal.
Function 11-12 are bound-constraint functions.
'''


def sphere(x):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: uni-modal; separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    value = 0
    for i in xrange(len(x)):
        value += x[i] * x[i]
    return value


def hyperEllipsoid(x):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: uni-modal; separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    value = 0
    for i in xrange(len(x)):
        value += pow(2, i) * (x[i] * x[i])
    return value
    

def rozenbrock(x):
    '''
    valid for any dimension, n>1
    constraints: unconstrained
    type: uni-modal; non-separable
    initial upper bound = 30, initial lower bound = -30
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(1,1,...,1)
    '''
    value = 0
    for i in xrange(len(x) - 1):
        value += (1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2
    return value

n = 9
for i in xrange(10):
    sphereOpt = DE(sphere, ([-100]*n,[100]*n), multiprocessing=False)
    sphereOpt.optimise()
    print i+1