from tests import TestDifferentialEvolution
import numpy

'''
This file contains the 20 test functions given in Appendix A of Storn and
Price's "Differential Evolution", implemented using the
TestDifferentialEvolution class.
'''

class SphereDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: uni-modal; separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            value += x[i] * x[i]
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([100] * self.dimensionality)
        return -1 * bound, bound

class HyperEllipsoidDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: uni-modal; separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            value += pow(2, i) * (x[i] * x[i])
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([100] * self.dimensionality)
        return -1 * bound, bound

class RozenbrockDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>1
    constraints: unconstrained
    type: uni-modal; non-separable
    initial upper bound = 30, initial lower bound = -30
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(1,1,...,1)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality - 1):
            value += (1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([30] * self.dimensionality)
        return -1 * bound, bound

class SchwefelDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: uni-modal; non-separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        a, value = 0, 0
        for i in xrange(self.dimensionality):
            a += x[i]
            value += (a * a)
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([100] * self.dimensionality)
        return -1 * bound, bound

class NeumaierDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>1
    constraints: unconstrained
    function type: multi-modal(?), one global optimum(?); non-separable
    initial upper bound = n*n, initial lower bound = -n*n
    value-to-reach = f(x*)+.5
    f(x*)= -n(n+4)(n-1)/6; x*[j]=(j+1)(n-j), j=0,1,...,n-1
    '''
    def cost(self, x):
        value = (x[0]-1) * (x[0]-1)
        for i in xrange(1, self.dimensionality):
            value += (x[i]-1) * (x[i]-1) - (x[i] * x[i-1])
        return value

    def get_bounding_vectors(self):
        d = self.dimensionality
        self.value_to_reach = ((-d * (d+4) * (d-1)) / 6.0) + 10**-6
        bound = numpy.array([d**2] * d)
        return -1 * bound, bound