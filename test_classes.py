import numpy
from differential_evolution import DifferentialEvolution

class NDimensionalRozenbrockDifferentialEvolution(DifferentialEvolution):
    '''
    Extends the DifferentialEvolution class to handle the problem
    of finding the minimum point in an N-dimensional Rozenbrock function.
    '''
    def __init__(self, n):
        self.n = n
        super(NDimensionalRozenbrockDifferentialEvolution, self).__init__()

    def cost(self, vector):
        '''
        In this case, we wish to minimise a Rozenbrock function in n dimensions
        '''
        value = 0
        for i in xrange(self.n - 1):
            value += (1-vector[i])**2 + 100*(vector[i+1] - vector[i]**2)**2
        return value

    def get_bounding_vectors(self):
        '''
        Get lower and upper bounds for the initial trial population
        '''
        bound = numpy.array([2] * self.n)
        return -1 * bound, bound


class EasomDifferentialEvolution(DifferentialEvolution):
    '''
    Extends the DifferentialEvolution class to handle the problem
    of finding the minimum point in a 2D Easom function.
    '''
    def cost(self, vector):
        '''
        In this case, we wish to minimise a 2D Easom function
        '''
        x, y = vector[0], vector[1]
        pi, cos, sin, exp = numpy.pi, numpy.cos, numpy.sin, numpy.exp
        exponent = -1 * ((x-pi)**2 +  (y-pi)**2)
        return -cos(x)*cos(y)*exp(exponent)

    def get_bounding_vectors(self):
        '''
        Get lower and upper bounds for the initial trial population
        '''
        bound = numpy.array([100, 100])
        return -1 * bound, bound


class PolynomialDifferentialEvolution(DifferentialEvolution):
    '''
    Extends the DifferentialEvolution class to find the local or
    global minimum or maximum of a one-dimensional polynomial.
    '''
    def __init__(self, coefficients, bounds, minimise=True):
        '''
        Get the polynomial coefficients and limits of the equation.
        Also work out if this is a minimisation or maximisation problem.
        '''
        self.coefficients = coefficients
        self.bounds = bounds
        super(PolynomialDifferentialEvolution, self).__init__()
        self.absolute_bounds = True
        self.minimise = minimise

    def cost(self, vector):
        '''
        Calculate the value of the polynomial at x = vector[0]
        '''
        value = 0
        x = vector[0]
        for i, coefficient in enumerate(self.coefficients):
            power = len(self.coefficients) - i - 1
            value += coefficient * pow(x, power)
        if self.minimise:
            return value
        else:
            return -value

    def get_bounding_vectors(self):
        '''
        Get lower and upper bounds for the initial trial population
        '''
        return numpy.array([self.bounds[0]]), numpy.array([self.bounds[1]])

def test(n):
    problem = NDimensionalRozenbrockDifferentialEvolution(n=n)
    problem.verbosity = 0
    for i in xrange(10):
        solution, iterations = problem.solve()
        print 'Standard Solution %s: %s'%(i, solution)
    problem = VariantNDimensionalRozenbrockDifferentialEvolution(n=n)
    problem.verbosity = 0
    for i in xrange(10):
        solution, iterations = problem.solve()
        print 'Variant Solution %s: %s'%(i, solution)


problem = NDimensionalRozenbrockDifferentialEvolution(n=6)
# problem = EasomDifferentialEvolution()
# problem = PolynomialDifferentialEvolution([1,-10,21,40,-100],[-10,10], False)
problem.verbosity = 1
solution, iterations = problem.solve()
print '\nSolution:\n%s\n\nTotal iterations: %s'%(solution, iterations)

# test(12)