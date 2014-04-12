'''
valid for any dimension, n>0
constraints: bound constrained
type: multi-modal with one global minimum; separable
initial upper bound = 500, initial lower bound = -500
value-to-reach = f(x*)+.01
f(x*) = -418.983; x*=(s,s,...,s), s=420.968746
'''
import numpy

absoluteBounds = True
vtr = -418.983 + 0.01

def cost(x):
    value = 0
    d = len(x)
    for i in xrange(d):
        value -= x[i] * numpy.sin(numpy.sqrt(numpy.abs(x[i])))
    return value / float(d)

def getBounds(d):
    bound = numpy.array([500] * d)
    return -1 * bound, bound