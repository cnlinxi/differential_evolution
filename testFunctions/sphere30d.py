'''
valid for any dimension, n>0
constraints: unconstrained
type: uni-modal; separable
initial upper bound = 100, initial lower bound = -100
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(0,0,...,0)
'''
def cost(x):
    value = 0
    for i in range(len(x)):
        value += x[i] * x[i]
    return value
    
def getBounds():
    dimensions = 30
    return [-100] * dimensions, [100] * dimensions