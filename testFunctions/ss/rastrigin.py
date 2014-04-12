'''
valid for any dimension, n>0
constraints:  unconstrained
type: multi-modal with one global minimum; separable
initial upper bound = 5.12, initial lower bound = -5.12
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(0,0,...,0)
'''
import numpy

vtr = 10**-6

def cost(x):
    value = 0
    for i in xrange(len(x)):
        value += x[i]*x[i] - 10*numpy.cos(2*numpy.pi*x[i]) + 10
    return value

def getBounds(dimensions):
    bound = numpy.array([5.12] * dimensions)
    return -1 * bound, bound