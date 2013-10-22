# We rely on numpy for array/vector operations and advanced maths.
# Also use time information for logging
import numpy, datetime

class DEMutatorsMixin(object):
    '''
    Functions used in self-adaptive differential evolution to mutate three
    or more vectors into a trial vector.
    Under the self-adaptive scheme, the chance of each of these functions
    being used is governed by a variable probability distribution.

    Note this is a mixin - it requires some additional functions from the
    DifferentialEvolution class to work.

    Each of the mutator functions take an argument 'i', which is the current
    index in the trial population.
    '''
    def _n_m_e_r_i(self, n, maximum, minimum=0, not_equal_to=[]):
        '''
        Helper function to return N Mutually Exclusive Random Integers
        in the range [minimum, maximum). Optionally takes a 'not_equal_to'
        argument; a list of integers which will be excluded from the set.
        '''
        selected = set()
        while len(selected) < n:
            rand = numpy.random.randint(minimum, maximum)
            # No need to check if rand in selected as selected is a set.
            if rand not in not_equal_to:
                selected.add(rand)
        return tuple(selected)

    def absolute_bounds(self, mutant):
        '''
        Force a mutant vector to lie within the given boundaries.
        '''
        mutant = numpy.minimum(self.max_vector, mutant)
        mutant = numpy.maximum(self.min_vector, mutant)
        return mutant

    def get_leader_index(self):
        '''
        Get the index of the best-performing member of the population
        '''
        return min(xrange(len(self.costs)), key=self.costs.__getitem__)

    def basic_mutation(self, r0, r1, r2):
        '''
        Mutation helper function, called by all mutation types.
        Returns v2 + [f * (v1 - v0)], where v0, v1 and v2 are vectors in the
        population with indices r0, r1 and r2.
        '''
        v0, v1, v2 = self.population[r0], self.population[r1], self.population[r2]
        mutant = v2 + (self.f * (v1 - v0))
        return mutant

    def de_rand_1(self, i):
        '''
        'Classic' DE mutation - combine three random vectors.
        '''
        r0, r1, r2 = self._n_m_e_r_i(3, self.population_size, not_equal_to=[i])
        return self.basic_mutation(r0, r1, r2)

    def de_best_1(self, i):
        '''
        Variation on classic DE, using the best-so-far vector as v0
        '''
        r0 = self.get_leader_index()
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[r0,i])
        return self.basic_mutation(r0, r1, r2)

    def de_current_to_best_1(self, i):
        '''
        Hybrid of de/rand/1 and de/best/1.
        '''
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[i])
        mutant = self.basic_mutation(i, r1, r2)
        vi, vbest = self.population[i], self.population[self.get_leader_index()]
        mutant += self.f * (vbest - vi)
        return mutant

    def de_rand_2(self, i):
        '''
        Like de/rand/1, but adds two random scaled vectors.
        '''
        r0, r1, r2, r3, r4 = self._n_m_e_r_i(5, self.population_size, not_equal_to=[i])
        mutant = self.basic_mutation(i, r1, r2)

    def de_best_2(self, i):
        '''
        Like de/best/1, but adds two random scaled vectors.
        '''




class DifferentialEvolution(DEMutatorsMixin):
    '''
    A class-based approach to the Differential Evolution (DE) problem.
    This class should be subclassed for use in specific problems.

    DE is an optimisation algorithm that minimises an n-dimensional
    cost function using one or more mutation functions.
    '''
    def __init__(self):
        '''
        Specify the problem default parameters as attributes here.
        Any of these attributes can be overriden.
        '''
        # Mutation scaling factor. Recommended 0.5 < f < 1.2
        self.f = 0.85
        # Crossover factor (see def crossover)
        self.c = 0.85
        # Number of iterations before the program terminates regardless
        # of convergence
        self.max_iterations = 10**4
        # Number of decimal places to which solutions are given.
        self.decimal_precision = 3
        # Maximum standard deviation of the population for the solution to be
        # considered converged
        self.convergence_std = 10**(-1*(self.decimal_precision + 1))
        # Get the min and max vectors
        self.min_vector, self.max_vector = self.get_bounding_vectors()
        assert len(self.min_vector) == len(self.max_vector)
        # Having checked min_vector and max_vector are the same length, arbitrarily
        # take min_vector to establish the number of dimensions in the problem.
        self.dimensionality = len(self.min_vector)
        # Are the boundaries of the problem absolute, i.e. are mutations outside
        # the bounding vectors banned?
        self.absolute_bounds = False
        # Default value for population size is 5 times the
        # problem dimensionality (2D = 10, 3D = 15, etc.).
        # I found a power law to be slightly
        # more reliable at low dim. and faster at high dim.
        self.population_size = int(12 * (self.dimensionality) ** 0.4)
        # Select logging amount. 0=silent, 1=basic, 2=verbose.
        self.verbosity = 2

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
            # A random vector in the range -0.5 - 0.5
            vector = numpy.random.rand(self.dimensionality) - 0.5
            # Manipulate it so that it meets the specified min/max conditions
            vector *= range
            vector += mean
            # Add the fully-constructed vector to the population
            population.append(numpy.array(vector))
        return population

    def mutation(self, v1, v2, v3):
        '''
        Creates a vector mutation by calculating v3 + [f * (v2 - v1)],
        in accordance with Price and Storn's 'Differential Evolution' approach.
        f, a real number, is known as the scaling factor.
        If absoulte bounds are set, make sure that the vector lies within the
        allowed boundaries.
        '''
        mutant = v3 + (self.f * (v2 - v1))
        if self.absolute_bounds:
            mutant = numpy.minimum(self.max_vector, mutant)
            mutant = numpy.maximum(self.min_vector, mutant)
        return mutant

    def crossover(self, v1, v2):
        '''
        Creates a trial vector by crossing v1 with v2 to create v3.
        The probability of a v2 element being selected over a v1 element is c,
        the crossover factor. There also exists an 'i_rand' to guarantee that
        at least one mutant value is chosen in the crossover.
        '''
        v3 = []
        i_rand = numpy.random.randint(self.dimensionality)
        random_array = numpy.random.rand(self.dimensionality)
        for i, random_number in enumerate(random_array):
            if random_number > self.c and i != i_rand:
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

    def basic_logging(self):
        '''
        This basic logging information is always shown unless verbosity is
        set to 0 (silent).
        '''
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%A, %d %B, %Y %I:%M%p")
        print '\ndifferential_evolution.py'
        print '\n2013 Blake Hemingway'
        print 'The University of Sheffield'
        print '\nRun started on %s'%(time_string)
        print '\nSolution parameters:\n'
        print 'Scaling factor:  \t%s'%(self.f)
        print 'Crossover factor:\t%s'%(self.c)
        print 'Convergence std: \t%s'%(self.convergence_std)
        print 'Population size: \t%s'%(self.population_size)
        print ''
        if self.verbosity > 1:
            self.col_spacing = self.dimensionality * (self.decimal_precision + 5)
            cs = self.col_spacing
            print ('Iteration'.ljust(cs/2) + 'Mean'.ljust(cs) +
                'Standard Deviation'.ljust(cs) + 'Current Best'.ljust(cs/2))

    def log_solution(self, iteration):
        '''
        Log data about the progress of the solution such as mean,
        standard deviation and current leader. Only called if the
        'verbose_logging' parameter is set
        '''
        population_stack = numpy.column_stack(self.population)
        mean = numpy.mean(population_stack, axis=1)
        mean = numpy.round(mean, self.decimal_precision)
        std = numpy.std(population_stack, axis=1)
        std = numpy.round(std, self.decimal_precision)
        best = numpy.round(min(self.costs), self.decimal_precision)
        cs = self.col_spacing
        print '%s%s%s%s'%(str(iteration).ljust(cs/2), str(mean).ljust(cs),
            str(std).ljust(cs), str(best).ljust(cs/2))

    def de(self):
        '''
        Two generation, unthreaded version
        '''
        # Loop over the vectors in the population
        trial_population = []
        for i, parent in enumerate(self.population):
            # Select three random vectors from the population (not i)
            r0, r1, r2 = self._n_mutually_exclusive_random_integers(
                3, self.population_size, not_equal_to = i)
            # Go through the mutation/crossover process
            mutant = self.mutation(r0, r1, r2)
            trial = self.crossover(parent, mutant)
            trial_population.append(trial)
        for i, trial in enumerate(trial_population):
            # Select the winner and update the population/costs.
            self.crown_tournament_victor(trial, i)

    def solve(self):
        '''
        This is the main function which initiates the solution process
        and returns a final answer.
        '''
        self.population = self.initialise_population()
        self.costs = [self.cost(vector) for vector in self.population]
        # Initialise the logging process
        if self.verbosity != 0:
            self.basic_logging()
        # Start iterating.
        for i in xrange(self.max_iterations):
            # Evolve the next generation
            self.de()
            # If logging, show output
            if self.verbosity > 1:
                self.log_solution(i+1)
            # Check for solution convergence
            convergence = self.check_for_convergence()
            if convergence:
                # Return the solution that minimises the cost function
                victor_index = min(xrange(len(self.costs)),
                    key=self.costs.__getitem__)
                victor = numpy.round(self.population[victor_index],
                    self.decimal_precision)
                return victor, i+1
        # If we get to here, we haven't achieved convergence. Raise an error.
        raise Exception('The solution did not converge')

class SingleGenerationMixin(object):
    '''
    This is the memory-saving DE variant described in Section 5.2.4 of Price
    and Storn's 'Differential Evolution'. There is only a single population
    and there is no clear distinction between generations.
    '''
    def de(self):
        '''
        Single generation, memory saving version
        '''
        # Loop over the vectors in the population
        for j in xrange(self.population_size):
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