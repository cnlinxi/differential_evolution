'''
valid for any dimension, n>1
constraints: bound constrained
type: multi-modal with one global minimum; non-separable
initial upper bound = 511, initial lower bound = -512
value-to-reach = f(x*)+.01
f(x*) = -511.708; x*=(-512,-512,...,-512)
'''
import numpy

vtr = -511.708 + 0.01
absoluteBounds = True

def cost(x):
    value = 0
    d = len(x)
    for i in xrange(d):
        j = (i+1) % d
        a = numpy.sqrt(numpy.abs(x[j] + 1 - x[i]))
        b = numpy.sqrt(numpy.abs(x[j] + 1 + x[i]))
        value += (x[i] * numpy.sin(a) * numpy.cos(b) +
                (x[j] + 1) * numpy.cos(a) * numpy.sin(b))
    return value / float(d)

def getBounds(d):
    bound = numpy.array([512] * d)
    return -1 * bound, bound