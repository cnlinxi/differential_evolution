'''
valid for any dimension, n>0
constraints: unconstrained
function type: multi-modal, one global optimum; non-separable
initial upper bound = 600, initial lower bound = -600
value-to-reach = f(x*)+1.0e-6
f(x*)= 0.0; x*=(0,0,...,0)
'''
import numpy

vtr = 10**-6

def cost(x):
    value = 0
    prod = 1.0
    for i in xrange(len(x)):
        value += x[i]*x[i]
        prod *= numpy.cos(x[i]/numpy.sqrt(i+1))
    return (value / 4000.0) - prod + 1

def getBounds(dimensions):
    bound = numpy.array([600] * dimensions)
    return -1 * bound, bound