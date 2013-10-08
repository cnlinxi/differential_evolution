# We rely on numpy for array/vector operations and advanced maths.
import numpy

class DifferentialEvolution(object):
    '''
    A class-based approach to the Differential Evolution (DE) problem.
    This class should be subclassed for use in specific problems.

    DE is an optimisation algorithm that minimises an n-dimensional cost function.
    '''

    def __init__(self):
        '''
        Specify the problem default parameters as attributes here.
        Any of these attributes can be overriden.
        '''
        # Mutation scaling factor. Recommended 0.5 < f < 1.2
        self.f = 0.9
        # Crossover factor (see def crossover)
        self.c = 0.9
        # Number of iterations before the program terminates regardless
        # of convergence
        self.max_iterations = 10**4
        # Maximum standard deviation of the population for the solution to be
        # considered converged
        self.convergence_std = 10**-8
        # Number of decimal places to which solutions are given.
        # Default is None for no rounding.
        self.decimal_precision = None
        # Get the min and max vectors
        self.min_vector, self.max_vector = self.get_bounding_vectors()
        assert len(self.min_vector) == len(self.max_vector)
        # Having checked min_vector and max_vector are the same length, arbitrarily
        # take min_vector to establish the number of dimensions in the problem.
        self.dimensionality = len(self.min_vector)
        # Default value for population size is 5 times the
        # problem dimensionality (2D = 10, 3D = 15, etc.).
        # I found 10 times the square root of the dimensionality to be slightly
        # more reliable at low dim. and faster at high dim.
        self.population_size = int(10 * (self.dimensionality) ** 0.5)

    # The following two functions must be implemented in the subclass.
    def cost(self, vector):
        '''
        This function should return a single value (lower is better)
        against which 'vector' is judged.
        '''
        raise NotImplementedError

    def get_bounding_vectors(self):
        '''
        This function should return two vectors: A min_vector and max_vector.
        These give the upper and lower limits of the initial population
        coordinates.
        '''
        raise NotImplementedError

    def initialise_population(self):
        '''
        Creates n vectors of random numbers within the ranges specified by
        min_vector and max_vector.
        The size of the vectors will be equal to that of min_vector and
        max_vector (which must be of equal size to each other).
        '''
        # Calculate the mean and range (vectors) of min_vector and max_vector.
        # These values are used to scale the random population.
        mean = numpy.mean(numpy.column_stack((self.min_vector, self.max_vector)), axis=1)
        range = self.max_vector - self.min_vector
        population = [] # A blank container to hold the population before returning.
        for i in xrange(self.population_size):
            vector = []
            for j in xrange(self.dimensionality):
                # Get a random number in the range -0.5 < r < 0.5
                random_number = numpy.random.rand() - 0.5
                # Manipulate it so that it meets the specified min/max conditions
                random_number = (random_number * range[j]) + mean[j]
                # Add it to vector i
                vector.append(random_number)
            # Add the fully-constructed vector to the population
            population.append(numpy.array(vector))
        return population

    def compute_all_costs(self):
        '''
        Called after initialise_population, this function picks up the costs
        for the whole population and returns them as a list
        '''
        costs = []
        for vector in self.population:
            costs.append(self.cost(vector))
        return costs

    def mutation(self, v1, v2, v3):
        '''
        Creates a vector mutation by calculating v3 + [f * (v2 - v1)],
        in accordance with Price and Storn's 'Differential Evolution' approach.
        f, a real number, is known as the scaling factor.
        '''
        return v3 + (self.f * (v2 - v1))

    def crossover(self, v1, v2):
        '''
        Creates a trial vector by crossing v1 with v2 to create v3.
        The probability of a v2 element being selected over a v1 element is c,
        the crossover factor. There also exists an 'i_rand' to guarantee that
        at least one mutant value is chosen in the crossover.
        '''
        v3 = []
        i_rand = numpy.random.randint(self.dimensionality)
        for i in xrange(self.dimensionality):
            random_number = numpy.random.rand()
            if random_number > self.c and i!=i_rand:
                v3.append(v1[i])
            else:
                v3.append(v2[i])
        return numpy.array(v3)

    def crown_tournament_victor(self, challenger, population_index):
        '''
        Gets the vector and associated cost of whichever vector minimises the
        cost function. The contest is between a 'challenger' vector and a
        'parent' vector in the population with index = population_index.
        '''
        challenger_cost = self.cost(challenger)
        if challenger_cost < self.costs[population_index]:
            self.population[population_index] = challenger
            self.costs[population_index] = challenger_cost

    def check_for_convergence(self):
        '''
        Returns True if the standard deviation of the population is below the
        specified value in all dimensions.
        '''
        std = numpy.std(numpy.column_stack(self.population), axis=1)
        if max(std) < self.convergence_std:
            return True
        else:
            return False

    def solve(self):
        '''
        This is the main function which implements the differential evolution
        loop.
        '''
        self.population = self.initialise_population()
        self.costs = self.compute_all_costs()
        # Start iterating.
        for i in xrange(self.max_iterations):
            # Loop over the vectors in the population
            for j in xrange(self.population_size):
                parent_cost = self.costs[j]
                # Select three random vectors from the population (not j)
                selected = []
                while len(selected) < 3:
                    randint = numpy.random.randint(self.population_size)
                    if randint != j:
                        selected.append(self.population[randint])
                # Go through the mutation/crossover process
                mutant = self.mutation(*selected)
                trial = self.crossover(self.population[j], mutant)
                # Select the winner and update the population/costs.
                self.crown_tournament_victor(trial, j)
            # Check for solution convergence
            convergence = self.check_for_convergence()
            if convergence:
                mean = numpy.mean(numpy.column_stack(self.population), axis=1)
                if self.decimal_precision:
                    mean = numpy.round(mean, self.decimal_precision)
                return (mean), i
        # If we get to here, we haven't achieved convergence. Raise an error.
        raise Exception('The solution did not converge')


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
        bound = numpy.array([3] * self.n)
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

problem = NDimensionalRozenbrockDifferentialEvolution(n=20)
# problem = EasomDifferentialEvolution()
problem.decimal_precision = 3
problem.c = 0.7
solution, iterations = problem.solve()
print '\nSolution:\n%s\n\nTotal iterations: %s'%(solution, iterations)