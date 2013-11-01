from tests import TestDifferentialEvolution
import numpy

'''
This file contains 12 of the 20 test functions given in Appendix A of
Storn and Price's "Differential Evolution", implemented using the
TestDifferentialEvolution class.
Functions 1-5 are unimodal.
Functions 6-10 are multimodal.
Function 11-12 are bound-constraint functions.
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

class SchwefelRidgeDifferentialEvolution(TestDifferentialEvolution):
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

class AckleyDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    function type: multi-modal, one global optimum
    initial upper bound = 30, initial lower bound = -30
    value-to-reach = f(x*)+1.0e-6
    f(x*)= 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        sum1, sum2 = 0, 0
        for i in xrange(self.dimensionality):
            sum1 += x[i]*x[i]
            sum2 += numpy.cos(2.*numpy.pi*x[i])
        sum1 = sum1 / float(self.dimensionality)
        sum2 = sum2 / float(self.dimensionality)
        return -20 * numpy.exp(-0.2*numpy.sqrt(sum1))-numpy.exp(sum2) + 20 + numpy.exp(1)

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([30] * self.dimensionality)
        return -1 * bound, bound

class GriewangkDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    function type: multi-modal, one global optimum; non-separable
    initial upper bound = 600, initial lower bound = -600
    value-to-reach = f(x*)+1.0e-6
    f(x*)= 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        value = 0
        prod = 1.0
        for i in xrange(self.dimensionality):
            value += x[i]*x[i]
            prod *= numpy.cos(x[i]/numpy.sqrt(i+1))
        return (value / 4000.0) - prod + 1

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([600] * self.dimensionality)
        return -1 * bound, bound

class RastriginDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints:  unconstrained
    type: multi-modal with one global minimum; separable
    initial upper bound = 5.12, initial lower bound = -5.12
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            value += x[i]*x[i] - 10*numpy.cos(2*numpy.pi*x[i]) + 10
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([5.12] * self.dimensionality)
        return -1 * bound, bound

class SalomonDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: multi-modal with one global minimum; non-separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(0,0,...,0)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            value += x[i]*x[i]
        a = numpy.sqrt(value)
        return -numpy.cos(2*numpy.pi*a) + 0.1*a + 1

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([100] * self.dimensionality)
        return -1 * bound, bound

class WhitleyDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: unconstrained
    type: multi-modal with one global minimum; non-separable
    initial upper bound = 100, initial lower bound = -100
    value-to-reach = f(x*)+1.0e-6
    f(x*) = 0.0; x*=(1,1,...1)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            for j in xrange(self.dimensionality):
                a = x[i]*x[i] - x[j]
                b = 1 - x[j]
                v = (100 * a * a) + (b * b)
                value += (v*v/4000.0) - numpy.cos(v) + 1
        return value

    def get_bounding_vectors(self):
        self.value_to_reach = 10**-6
        bound = numpy.array([100] * self.dimensionality)
        return -1 * bound, bound

class SchwefelDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>0
    constraints: bound constrained
    type: multi-modal with one global minimum; separable
    initial upper bound = 500, initial lower bound = -500
    value-to-reach = f(x*)+.01
    f(x*) = -418.983; x*=(s,s,...,s), s=420.968746
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            value -= x[i] * numpy.sin(numpy.sqrt(numpy.abs(x[i])))
        return value / float(self.dimensionality)

    def get_bounding_vectors(self):
        self.absolute_bounds = True
        self.value_to_reach = -418.983 + 0.01
        bound = numpy.array([500] * self.dimensionality)
        return -1 * bound, bound

class RanaDifferentialEvolution(TestDifferentialEvolution):
    '''
    valid for any dimension, n>1
    constraints: bound constrained
    type: multi-modal with one global minimum; non-separable
    initial upper bound = 511, initial lower bound = -512
    value-to-reach = f(x*)+.01
    f(x*) = -511.708; x*=(-512,-512,...,-512)
    '''
    def cost(self, x):
        value = 0
        for i in xrange(self.dimensionality):
            j = (i+1) % self.dimensionality
            a = numpy.sqrt(numpy.abs(x[j] + 1 - x[i]))
            b = numpy.sqrt(numpy.abs(x[j] + 1 + x[i]))
            value += (x[i] * numpy.sin(a) * numpy.cos(b) +
                    (x[j] + 1) * numpy.cos(a) * numpy.sin(b))
        return value / float(self.dimensionality)

    def get_bounding_vectors(self):
        self.absolute_bounds = True
        self.value_to_reach = -511.708 + 0.01
        bound = numpy.array([512] * self.dimensionality)
        return -1 * bound, bound