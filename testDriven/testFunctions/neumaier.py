'''
valid for any dimension, n>1
constraints: unconstrained
function type: multi-modal(?), one global optimum(?); non-separable
initial upper bound = n*n, initial lower bound = -n*n
value-to-reach = f(x*)+.5
f(x*)= -n(n+4)(n-1)/6; x*[j]=(j+1)(n-j), j=0,1,...,n-1
'''
def cost(x):
    value = (x[0]-1) * (x[0]-1)
    for i in xrange(1, len(x)):
        value += (x[i]-1) * (x[i]-1) - (x[i] * x[i-1])
    return value

def getBounds(d):
    return [-(d**2)] * d, [d**2] * d
    
def getVtr(d):
    return ((-d * (d+4) * (d-1)) / 6.0) + 10**-6