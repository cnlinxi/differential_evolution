'''
valid for any dimension, n>0
constraints: unconstrained
type: multi-modal with one global minimum; non-separable
initial upper bound = 100, initial lower bound = -100
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(0,0,...,0)
'''
import numpy

vtr = 10**-6

def cost(x):
    value = 0
    for i in xrange(len(x)):
        value += x[i]*x[i]
    a = numpy.sqrt(value)
    return -numpy.cos(2*numpy.pi*a) + 0.1*a + 1

def getBounds(dimensions):
    bound = numpy.array([100] * dimensions)
    return -1 * bound, bound