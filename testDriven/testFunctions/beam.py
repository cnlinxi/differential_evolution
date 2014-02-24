'''
valid for any dimension, n>1
constraints: bound constrained
type: unimodal with one global minimum; non-seperable
initial upper bound = 10, initial lower bound = 0
value-to-reach = f(x*)+.01
f(x*) = 175
'''
absoluteBounds = True
sequential = True

def cost(x):
    values = [(x[0]**4)/8.0, ((10-x[-1])**4)/8.0]
    for i in xrange(1, len(x)):
        values.append(((x[i] - x[i-1])**4)/384.0)
    return max(values)
    
def getBounds(dimensions):
    return [0] * dimensions, [10] * dimensions