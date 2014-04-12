'''
valid for any dimension, n>0
constraints: unconstrained
type: uni-modal; separable
initial upper bound = 100, initial lower bound = -100
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(0,0,...,0)
'''
vtr = 10**-6 

def cost (x):
    value = 0
    for i in xrange(len(x)):
        value += pow(2, i) * (x[i] * x[i])
    return value

def getBounds(dimensions):
    return [-100] * dimensions, [100] * dimensions