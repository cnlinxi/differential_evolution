# We rely on numpy for array/vector operations and advanced maths.
# Also use time information for logging
import numpy, datetime

class NotConvergedException(Exception):
    '''
    Exception raised when a solution is not found
    '''
    pass

class DifferentialEvolution(object):
    '''
    A class-based approach to the Differential Evolution (DE) problem.
    This class should be subclassed for use in specific problems.

    DE is an optimisation algorithm that minimises an n-dimensional
    cost function using one or more mutation functions.

    This is a basic version, as set out by Storn and Price in
    'Differential Evolution - A Practical Approach to Global Optimisation'.

    There are three tiers of functions:
    - Top-level functions control the program flow and manipulate object variables.
    - Second-level functions are the tools drawn upon to adjust the 'flavour' of
      top-level functions. e.g. de_rand_1 to mutation. They do not manipulate
      object variables.
    - Third-level functions are basic helper functions. Their names are prefixed
      with a single underscore.
    '''
    def __init__(self):
        '''
        Specify the problem default parameters as attributes here.
        Any of these attributes can be overriden.
        '''
        # Start by listing our (discrete) choices
        self.mutators = {
            'de/rand/1/bin': self.de_rand_1,
            'de/best/1/bin': self.de_best_1,
            'de/current_to_best/1/bin': self.de_current_to_best_1,
            'de/rand/2/bin': self.de_rand_2,
            'de/best/2/bin': self.de_best_2,
        }
        self.base_vector_selectors = {
            'random': self.random_base_vector_selection,
            'permuted': self.permuted_base_vector_selection,
            'offset': self.random_offset_base_vector_selection,
        }
        self.convergence_functions = {
            'std': self.std_convergence,
            'vtr': self.vtr_convergence,
        }
        # Selection of base vector scheme
        self.base_vector_selection_scheme = 'random'
        # Select algorithm used for mutation
        self.mutation_scheme = 'de/rand/1/bin'
        # Select convergence function
        self.convergence_function = 'std'
        # Mutation scaling factor. Recommended 0.5 < f < 1.2
        self.f = 0.85
        # Crossover factor (see def crossover)
        self.c = 0.85
        # Number of iterations before the program terminates regardless
        # of convergence
        self.max_iterations = 2000
        # Number of decimal places to which solutions are given.
        self.decimal_precision = 3
        # Maximum standard deviation of the population for the solution to be
        # considered converged. Only used if convergence_function = std_convergence
        self.convergence_std = 10**(-1*(self.decimal_precision + 1))
        # Value to reach. Only used if convergence_function = vtr_convergence
        self.value_to_reach = 0
        # Are the boundaries of the problem absolute, i.e. are mutations outside
        # the bounding vectors banned?
        self.absolute_bounds = False
        # Get the min and max vectors
        self.min_vector, self.max_vector = self.get_bounding_vectors()
        assert len(self.min_vector) == len(self.max_vector)
        # Having checked min_vector and max_vector are the same length, arbitrarily
        # take min_vector to establish the number of dimensions in the problem.
        self.dimensionality = len(self.min_vector)
        # Default value for population size is 5 times the
        # problem dimensionality (2D = 10, 3D = 15, etc.).
        # I found a power law to be slightly
        # more reliable at low dim. and faster at high dim.
        self.population_size = int(10 * (self.dimensionality) ** 0.5)
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

    # The following functions concern population initialisation.
    def generate_random_population(self, mean, range):
        '''
        This function creates a randomly-distributed initial population.
        Halton or Gaussian distributions could also be used.
        '''
        population = [] # A blank container to hold the population before returning.
        for i in xrange(self.population_size):
            # A random vector in the range -0.5 - 0.5
            vector = numpy.random.rand(self.dimensionality) - 0.5
            # Manipulate it so that it meets the specified min/max conditions
            vector *= range
            vector += mean
            # Add the fully-constructed vector to the population
            population.append(vector)
        return population

    def initialise_population(self):
        '''
        Creates n vectors within the ranges specified by
        min_vector and max_vector.
        The size of the vectors will be equal to that of min_vector and
        max_vector (which must be of equal size to each other).
        '''
        # Calculate the mean and range (vectors) of min_vector and max_vector.
        # These values are used to scale the population.
        mean = numpy.mean(numpy.column_stack((self.min_vector, self.max_vector)), axis=1)
        range = self.max_vector - self.min_vector
        return self.generate_random_population(mean, range)

    # The following second-level functions are used during mutation to
    # select the base vector, v0.
    def random_base_vector_selection(self):
        '''
        Base vectors are selected with a random and independent probability.
        The only condition is that randoms[i] != i
        '''
        randoms = [0] # Arbitrary initialisation to imitate a do-while loop
        while any([randoms[i] == i for i in randoms]):
            randoms = numpy.random.randint(self.population_size, size=self.population_size)
        return randoms

    def permuted_base_vector_selection(self):
        '''
        Base vectors are selected randomly but with dependent probability.
        Each vector is used as v0 only once per generation.
        '''
        randoms = numpy.random.permutation(self.population_size)
        while any([randoms[i] == i for i in randoms]):
            numpy.random.shuffle(randoms)
        return randoms

    def random_offset_base_vector_selection(self):
        '''
        Base vectors are selected at a random but unchanging offset from the
        parent vector.
        '''
        random_offset = numpy.random.randint(self.population_size)
        return [(i + random_offset) % self.population_size for i in xrange(self.population_size)]

    # The following functions concern mutation operations.
    # They are self-adaptive ready, but there is no self-adaptive logic here.
    def _n_m_e_r_i(self, n, maximum, minimum=0, not_equal_to=[]):
        '''
        Helper function to return N Mutually Exclusive Random Integers (nmeri)
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

    def enforce_absolute_bounds(self, mutant):
        '''
        Force a mutant vector to lie within the given boundaries.
        '''
        mutant = numpy.minimum(self.max_vector, mutant)
        mutant = numpy.maximum(self.min_vector, mutant)
        return mutant

    def get_best_vector_index(self):
        '''
        Get the index of the best-performing member of the population
        '''
        return min(xrange(len(self.costs)), key=self.costs.__getitem__)

    def basic_mutation(self, v0, v1, v2, f):
        '''
        Mutation helper function, called by all mutation types.
        Returns v0 + [f * (v1 - v2)], where v0, v1 and v2 are vectors
        '''
        return v0 + (f * (v1 - v2))

    def de_rand_1(self, i, f, r0):
        '''
        'Classic' DE mutation - combine three random vectors.
        '''
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[i, r0])
        v0, v1, v2 = self.population[r0], self.population[r1], self.population[r2]
        return self.basic_mutation(v0, v1, v2, f)

    def de_best_1(self, i, f, r0):
        '''
        Variation on classic DE, using the best-so-far vector as v0.
        r0 is allowed as an argument for consistency, but is not used.
        '''
        r_best = self.get_best_vector_index()
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[i, r_best])
        v0, v1, v2 = self.population[r_best], self.population[r1], self.population[r2]
        return self.basic_mutation(v0, v1, v2, f)

    def de_current_to_best_1(self, i, f, r0):
        '''
        Hybrid of de/rand/1 and de/best/1. r0 is again ignored.
        '''
        r_best = self.get_best_vector_index()
        vi, v_best = self.population[i], self.population[r_best]
        current_to_best = self.basic_mutation(vi, v_best, vi, f)
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[i, r_best])
        v1, v2 = self.population[r1], self.population[r2]
        return self.basic_mutation(current_to_best, v1, v2, f)

    def de_rand_2(self, i, f, r0):
        '''
        Like de/rand/1, but adds two random scaled vectors.
        '''
        r1, r2, r3, r4 = self._n_m_e_r_i(4, self.population_size, not_equal_to=[i, r0])
        v0, v1, v2 = self.population[r0], self.population[r1], self.population[r2]
        mutant = self.basic_mutation(v0, v1, v2, f)
        v3, v4 = self.population[r3], self.population[r4]
        return self.basic_mutation(mutant, v3, v4, f)

    def de_best_2(self, i, f, r0):
        '''
        Like de/best/1, but adds two random scaled vectors.
        '''
        r_best = self.get_best_vector_index()
        r1, r2, r3, r4 = self._n_m_e_r_i(4, self.population_size, not_equal_to=[i, r_best])
        v0, v1, v2 = self.population[r_best], self.population[r1], self.population[r2]
        mutant = self.basic_mutation(v0, v1, v2, f)
        v3, v4 = self.population[r3], self.population[r4]
        return self.basic_mutation(mutant, v3, v4, f)

    def mutation(self):
        '''
        Mutate all vectors in the population.
        Yields an iterator of mutants.

        In normal DE, only a single scheme and a single f value are used.
        '''
        base_vector_indices = self.base_vector_selection_scheme()
        for i, vi in enumerate(self.population):
            mutant = self.mutation_scheme(i, self.f, base_vector_indices[i])
            if self.absolute_bounds:
                mutant = self.enforce_absolute_bounds(mutant)
            yield mutant

    # The following function performs the crossover operation.
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

    # One of these functions is called at the end of each generation. The
    # second-tier convergence functions given here are selectable termination
    # criteria.
    def std_convergence(self):
        '''
        Returns True if the standard deviation of the population is below the
        specified value in all dimensions.
        '''
        std = numpy.std(numpy.column_stack(self.population), axis=1)
        return max(std) < self.convergence_std

    def vtr_convergence(self):
        '''
        Returns True when the lowest function cost dips below a given value.
        (a Value To Reach)
        '''
        return min(self.costs) < self.value_to_reach

    # The following functions log progress. Whether one or all of them is called
    # depends on the verbosity setting.
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

    # This is the engine of the algorithm - the solve function calls
    # de and tournament to optimise the given cost function.
    def de(self):
        '''
        This function creates a trial population from an existing population
        using mutation and crossover operations.
        It returns a trial population as an iterator.
        '''
        mutants = self.mutation()
        trial_population = (self.crossover(self.population[i], mutant)
            for i, mutant in enumerate(mutants))
        return trial_population

    def tournament(self, trial_population):
        '''
        This function calculates the cost function for each trial vector.
        If the trial vector is more successful, it replaces its parent in
        the population.
        '''
        for i, trial in enumerate(trial_population):
            # Select the winner and update the population/costs.
            trial_cost = self.cost(trial)
            self.function_evaluations += 1
            if trial_cost < self.costs[i]:
                self.population[i] = trial
                self.costs[i] = trial_cost

    def solve(self):
        '''
        This is the main function which initiates the solution process
        and returns a final answer.
        '''
        # Get functions corresponding to strings by using them as keys in lookup
        if isinstance(self.base_vector_selection_scheme, basestring):
            self.base_vector_selection_scheme = \
                self.base_vector_selectors.get(self.base_vector_selection_scheme)
        if isinstance(self.mutation_scheme, basestring):
            self.mutation_scheme = self.mutators.get(self.mutation_scheme)
        if isinstance(self.convergence_function, basestring):
            self.convergence_function = \
                self.convergence_functions.get(self.convergence_function)
        # Initialise the solver
        self.population = self.initialise_population()
        self.costs = [self.cost(vector) for vector in self.population]
        self.function_evaluations = self.population_size
        # Initialise the logging process
        if self.verbosity != 0:
            self.basic_logging()
        # Start iterating.
        for i in xrange(self.max_iterations):
            # If logging, show output
            if self.verbosity > 1:
                self.log_solution(i+1)
            # Evolve the next generation
            trial_population = self.de()
            # Tournament-select the next generation
            self.tournament(trial_population)
            # Check for solution convergence
            convergence = self.convergence_function()
            if convergence:
                # Return the solution that minimises the cost function
                victor_index = self.get_best_vector_index()
                victor = numpy.round(self.population[victor_index],
                    self.decimal_precision)
                return victor, i+1, self.function_evaluations
        # If we get to here, we haven't achieved convergence. Raise an error.
        raise NotConvergedException('The solution did not converge')