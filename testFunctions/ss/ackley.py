'''
valid for any dimension, n>0
constraints: unconstrained
function type: multi-modal, one global optimum
initial upper bound = 30, initial lower bound = -30
value-to-reach = f(x*)+1.0e-6
f(x*)= 0.0; x*=(0,0,...,0)
'''
import numpy 

vtr =  10**-6

def cost(x):
    sum1, sum2 = 0, 0
    dimensions = len(x)
    for i in xrange(len(x)):
        sum1 += x[i]*x[i]
        sum2 += numpy.cos(2.*numpy.pi*x[i])
    sum1 = sum1 / float(dimensions)
    sum2 = sum2 / float(dimensions)
    return -20 * numpy.exp(-0.2*numpy.sqrt(sum1))-numpy.exp(sum2) + 20 + numpy.exp(1)

def getBounds(dimensions):
    bound = numpy.array([30] * dimensions)
    return -1 * bound, bound