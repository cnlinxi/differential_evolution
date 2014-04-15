'''
valid for any dimension, n>0
constraints: bound constrained
type: uni-modal; separable, many flat plateaus
initial upper bound = 5.12, initial lower bound = -5.12
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(<=-5, <=-5)
'''
import numpy 

def cost(x):
    return 1e-6 + (6 * 30) + sum(numpy.floor(a) for a in x)
    
def getBounds():
    dimensions = 30
    return [-5.12] * dimensions, [5.12] * dimensions
    
absoluteBounds = True