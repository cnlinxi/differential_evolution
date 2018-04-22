'''
valid for any dimension, n>1
constraints: unconstrained
type: uni-modal; non-separable
initial upper bound = 30, initial lower bound = -30
value-to-reach = f(x*)+1.0e-6
f(x*) = 0.0; x*=(1,1,...,1)
'''
def cost(x):
    value = 0
    for i in range(len(x) - 1):
        value += (1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2
    return value

def getBounds():
    dimensions = 10
    return [-30] * dimensions, [30] * dimensions