'''
valid for any dimension, n>0
constraints: unconstrained
type: multi-modal with one global minimum; non-separable
initial upper bound = 100, initial lower bound = -100
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(1,1,...1)
'''
import numpy

vtr = 10**-6

def cost(x):
    value = 0
    d = len(x)
    for i in xrange(d):
        for j in xrange(d):
            a = x[i]*x[i] - x[j]
            b = 1 - x[j]
            v = (100 * a * a) + (b * b)
            value += (v*v/4000.0) - numpy.cos(v) + 1
    return value

def getBounds(dimensions):
    bound = numpy.array([100] * dimensions)
    return -1 * bound, bound