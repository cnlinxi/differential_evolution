# We rely on numpy for array/vector operations and advanced maths.
import numpy
import datetime
import itertools
import sys
import os

class NotConvergedException(Exception):
    '''
    Exception raised when a solution is not found
    '''
    pass
    
'''
For backwards-compatibility with Python 2.4.
'''
def any(s):
    for v in s:
        if v:
            return True
    return False
    
def min_with_key(iterable, key):
    # Only tested with lists or tuples!
    try:
        return min(iterable, key=key)
    except TypeError:
        keyed_iterable = [key(i) for i in iterable]
        m = min(keyed_iterable)
        i = keyed_iterable.index(m)
        return iterable[i]
        
'''
A general form of the print function, to change the output
location in one line.
'''
def printer(text, output='stderr'):
    if output == 'stdout':
        print text
    elif output == 'stderr':
        sys.__stderr__.write(str(text) + os.linesep)
        sys.__stderr__.flush()

class DifferentialEvolution(object):
    '''
    A class-based approach to the Differential Evolution (DE) problem.
    This class should be subclassed for use in specific problems.

    DE is an optimisation algorithm that minimises an n-dimensional
    cost function using one or more mutation functions.

    This is a basic version, as set out by Storn and Price in
    'Differential Evolution - A Practical Approach to Global Optimisation'.
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
            'de/rand_then_best/1/bin': self.de_rand_then_best_1,
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
        self.f_randomisers = {
            'static': lambda f: f,  # Just return f when this is called.
            'dither': self.dither,
            'jitter': self.jitter,
        }
        self.c_randomisers = {
            'static': lambda c: c,  # Just return c when this is called.
            'dither': self.dither,
        }
        # Selection of base vector scheme
        self.base_vector_selection_scheme = 'random'
        # Select algorithm used for mutation
        self.mutation_scheme = 'de/rand/1/bin'
        # Select convergence function
        self.convergence_function = 'std'
        # Mutation scaling factor. Recommended 0.5 < f < 1.2
        self.f = 0.85
        # Select f distribution
        self.f_randomisation = 'static'
        # Crossover factor (see def crossover)
        self.c = 0.85
        # Select c distribution
        self.c_randomisation = 'static'
        # Number of generations before the program terminates regardless
        # of convergence
        self.max_generations = 2000
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
        min_vector, max_vector = self.get_bounding_vectors()
        self.min_vector = numpy.asarray(min_vector)
        self.max_vector = numpy.asarray(max_vector)
        assert len(self.min_vector) == len(self.max_vector)
        # Having checked min_vector and max_vector are the same length, arbitrarily
        # take min_vector to establish the number of dimensions in the problem.
        self.dimensionality = len(self.min_vector)
        # Add support for inequality constraints.
        # These should be tuples, e.g. ('lte', 5), or None.
        self.inequality_constraints = [None] * self.dimensionality
        # Check for 'phantom' dimensions that are constrained to a single value.
        self.phantom_indices = []
        for i, x in enumerate(self.max_vector-self.min_vector):
            if x==0:
                self.phantom_indices.append(i)
        # Default value for population size is 5 times the
        # problem dimensionality (2D = 10, 3D = 15, etc.).
        # I found a power law to be slightly
        # more reliable at low dim. and faster at high dim.
        self.population_size = int(11.69 * self.dimensionality**0.63)
        # Select logging amount. 0=silent, 1=basic, 2=verbose.
        self.verbosity = 2

    def _string_to_function_if_string(self, attr, dictionary):
        '''
        Utility to convert a string to a function by lookup in a
        dictionary, or leave the function alone if it is already a function,
        then set self.attr equal to that function.
        '''
        func_or_string = getattr(self, attr)
        if isinstance(func_or_string, basestring):
            func = dictionary[func_or_string]
        else:
            func = func_or_string
        setattr(self, attr, func)

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
            # Enforce any inequality constraints
            vector = self.enforce_constraints(vector)
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

    # The following functions are used during mutation to
    # select the base vector, v0.
    def random_base_vector_selection(self):
        '''
        Base vectors are selected with a random and independent probability.
        The only condition is that randoms[i] != i
        '''
        randoms = [0] # Arbitrary initialisation to imitate a do-while loop
        while any(randoms[i] == i for i in randoms):
            randoms = numpy.random.randint(self.population_size, size=self.population_size)
        return randoms

    def permuted_base_vector_selection(self):
        '''
        Base vectors are selected randomly but with dependent probability.
        Each vector is used as v0 only once per generation.
        '''
        randoms = numpy.random.permutation(self.population_size)
        while any(randoms[i] == i for i in randoms):
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

    def enforce_absolute_bounds(self, vector):
        '''
        Force a mutant vector to lie within the given boundaries.
        '''
        vector = numpy.minimum(self.max_vector, vector)
        vector = numpy.maximum(self.min_vector, vector)
        return vector
        
    def enforce_constraints(self, vector):
        '''
        Enforce any specified inequality or absolute bounds
        constrains on a vector.
        '''
        for i in xrange(len(vector)):
            inequality = self.inequality_constraints[i]
            if inequality is not None:
                mode, comparator_index = inequality
                i_value = vector[i]
                comparator_value = vector[comparator_index]
                # Reverse the values such that they obey the inequality.
                if (mode == 'gte' and i_value < comparator_value) or (mode == 'gt' and i_value <= comparator_value):
                    vector[i] = comparator_value
                    vector[comparator_index] = i_value
        if self.absolute_bounds:
            vector = self.enforce_absolute_bounds(vector)
        return vector

    def dither(self, x, sigma=0.2):
        '''
        Returns a scalar based on a normal distribution about x
        with a standard deviation of sigma.
        '''
        return numpy.random.normal(x, sigma)

    def jitter(self, x, sigma=0.2):
        '''
        Returns a vector based on a normal distribution about x
        with a standard deviation of sigma.
        '''
        return numpy.random.normal(x, sigma, self.dimensionality)

    def get_best_vector_index(self):
        '''
        Get the index of the best-performing member of the population
        '''
        return min_with_key(xrange(len(self.costs)), key=self.costs.__getitem__)

    def basic_mutation(self, v0, v1, v2, f):
        '''
        Mutation helper function, called by all mutation types.
        Returns v0 + [f * (v1 - v2)], where v0, v1 and v2 are vectors.
        f may be static, dithered or jittered.
        '''
        f = self.f_randomisation(f)
        metadata = {'f': f}
        return v0 + (f * (v1 - v2)), metadata

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
        current_to_best = self.basic_mutation(vi, v_best, vi, f)[0]
        r1, r2 = self._n_m_e_r_i(2, self.population_size, not_equal_to=[i, r_best])
        v1, v2 = self.population[r1], self.population[r2]
        return self.basic_mutation(current_to_best, v1, v2, f)

    def de_rand_2(self, i, f, r0):
        '''
        Like de/rand/1, but adds two random scaled vectors.
        Modified from Qin and Suganthan by using f/2.
        '''
        r1, r2, r3, r4 = self._n_m_e_r_i(4, self.population_size, not_equal_to=[i, r0])
        v0, v1, v2 = self.population[r0], self.population[r1], self.population[r2]
        mutant = self.basic_mutation(v0, v1, v2, f*0.5)[0]
        v3, v4 = self.population[r3], self.population[r4]
        return self.basic_mutation(mutant, v3, v4, f*0.5)

    def de_best_2(self, i, f, r0):
        '''
        Like de/best/1, but adds two random scaled vectors.
        Modified from Qin and Suganthan by using f/2.
        '''
        r_best = self.get_best_vector_index()
        r1, r2, r3, r4 = self._n_m_e_r_i(4, self.population_size, not_equal_to=[i, r_best])
        v0, v1, v2 = self.population[r_best], self.population[r1], self.population[r2]
        mutant = self.basic_mutation(v0, v1, v2, f*0.5)[0]
        v3, v4 = self.population[r3], self.population[r4]
        return self.basic_mutation(mutant, v3, v4, f*0.5)

    def de_rand_then_best_1(self, i, f, r0):
        '''
        If the generation is odd, use de_rand_1.
        If the generation is even, use de_best_1.
        '''
        if self.generation % 2:
            return self.de_rand_1(i, f, r0)
        else:
            return self.de_best_1(i, f, r0)

    def mutation(self):
        '''
        Mutate all vectors in the population.
        Yields an iterator of mutants with metadata.
        '''
        base_vector_indices = self.base_vector_selection_scheme()
        for i in xrange(self.population_size):
            mutant, metadata = self.mutation_scheme(i, self.f, base_vector_indices[i])
            mutant = self.enforce_constraints(mutant)
            yield mutant, metadata

    # The following function performs the crossover operation.
    def crossover(self, v1, v2, metadata):
        '''
        Creates a trial vector by crossing v1 with v2 to create v3.
        The probability of a v2 element being selected over a v1 element is c,
        the crossover factor. There also exists an 'i_rand' to guarantee that
        at least one mutant value is chosen in the crossover.
        '''
        c = self.c_randomisation(self.c)
        metadata['c'] = c
        v3 = []
        i_rand = numpy.random.randint(self.dimensionality)
        while i_rand in self.phantom_indices:
            i_rand = numpy.random.randint(self.dimensionality)
        random_array = numpy.random.rand(self.dimensionality)
        for i, random_number in enumerate(random_array):
            if random_number > self.c and i != i_rand:
                v3.append(v1[i])
            else:
                v3.append(v2[i])
        return numpy.array(v3), metadata

    # One of these functions is called at the end of each generation. The
    # convergence functions given here are selectable termination criteria.
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
        printer('\n2014 Blake Hemingway')
        printer('The University of Sheffield')
        printer('\nRun started on %s'%(time_string))
        printer('\nInitial solution parameters:\n')
        printer('Scaling factor:  \t%s'%(self.f))
        printer('Crossover factor:\t%s'%(self.c))
        printer('Population size: \t%s\n'%(self.population_size))

    def log_solution(self, generation):
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
        printer('\nAt generation %s:'%(generation))
        printer('Mean values: %s'%(str(mean)))
        printer('std values: %s'%(str(std)))
        printer('Best-so-far function value: %s\n\n'%(best))

    # Hooks - called at key points throughout the solution process.
    def solution_commencing_hook(self):
        '''
        This hook is called after the population is initialised
        but before the solver starts. Could be used to define
        instance variables used in later hooks.
        '''
        pass

    def tournament_complete_hook(self, trial_cost, parent_cost, metadata):
        '''
        This hook is called after each tournament.
        It could be used to perform postprocessing or logical (i.e. self-
        adaptive) operations.
        '''
        pass

    def generation_complete_hook(self):
        '''
        This hook is called after each generation.
        It could be used to perform postprocessing or logical (i.e. self-
        adaptive) operations.
        '''
        pass

    def solution_complete_hook(self, victor):
        '''
        This hook is called after a solution is successfully found.
        '''
        return (victor, self.generation, self.function_evaluations,
                self.population_size, None, None, None)

    # This is the engine of the algorithm - the solve function calls
    # de and tournament to optimise the given cost function.
    def de(self):
        '''
        This function creates a trial population from an existing population
        using mutation and crossover operations.
        It returns a trial population as an iterator.
        '''
        mutants_with_metadata = self.mutation()
        for i, (mutant, metadata) in enumerate(mutants_with_metadata):
            trial_vector, metadata = self.crossover(
                    self.population[i], mutant, metadata)
            yield trial_vector, metadata

    def tournament(self, trial_vector_index, trial_vector, metadata):
        '''
        This function calculates the cost function for each trial vector.
        If the trial vector is more successful, it replaces its parent in
        the population.
        '''
        trial_cost = self.cost(trial_vector)
        self.function_evaluations += 1
        parent_cost = self.costs[trial_vector_index]
        printer('The trial vector %s has an associated cost function of %s'%(
            str(trial_vector), str(trial_cost)))
        # Less than or equal is important to avoid stagnation
        # in quantised landscapes.
        if trial_cost <= parent_cost:
            printer('The trial cost was found to be lower than the parent cost of %s; therefore, the child vector has replaced its parent.'%(
                str(parent_cost)))
            self.population[trial_vector_index] = trial_vector
            self.costs[trial_vector_index] = trial_cost
        else:
            printer('The trial cost was found to be greater than the parent cost of %s; therefore, the parent vector remains.'%(
                str(parent_cost)))
        self.tournament_complete_hook(trial_cost, parent_cost, metadata)

    def solve(self):
        '''
        This is the main function which initiates the solution process
        and returns a final answer.
        '''
        # Get functions corresponding to strings by using them as keys in lookup
        self._string_to_function_if_string(
                'base_vector_selection_scheme', self.base_vector_selectors)
        self._string_to_function_if_string(
                'mutation_scheme', self.mutators)
        self._string_to_function_if_string(
                'convergence_function', self.convergence_functions)
        self._string_to_function_if_string(
                'f_randomisation', self.f_randomisers)
        self._string_to_function_if_string(
                'c_randomisation', self.c_randomisers)
        # Initialise the logging process
        if self.verbosity != 0:
            self.basic_logging()
        # Initialise the solver
        self.population = self.initialise_population()
        self.costs = [self.cost(vector) for vector in self.population]
        self.function_evaluations = self.population_size
        # Start iterating.
        self.solution_commencing_hook()
        for i in xrange(self.max_generations):
            self.generation = i+1
            # If logging, show output
            if self.verbosity > 1:
                self.log_solution(self.generation)
            # Evolve the next generation
            trial_population_with_metadata = self.de()
            # Tournament-select the next generation
            for j, (trial_vector, metadata) in enumerate(trial_population_with_metadata):
                self.tournament(j, trial_vector, metadata)
            self.generation_complete_hook()
            # Check for solution convergence
            convergence = self.convergence_function()
            if convergence:
                # Return the solution that minimises the cost function
                victor_index = self.get_best_vector_index()
                victor = numpy.round(self.population[victor_index],
                    self.decimal_precision)
                return self.solution_complete_hook(victor)
        # If we get to here, we haven't achieved convergence. Raise an error.
        raise NotConvergedException('The solution did not converge', min(self.costs))


class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
    '''
    DE, modified such that there is no need to preselect c or f_randomisation
    '''
    def roulette_wheel(self, names, success_history, population_protection=0.02):
        '''
        Redistribute a roulette-wheel selection system based on a success
        history. Optionally protect 'weaker' entities through
        population_protection to prevent extinction.
        '''
        success_history_length = float(len(success_history))
        thresholds = numpy.zeros(len(names) + 1)
        for i, name in enumerate(names):
            success_count = success_history.count(name)
            success_proportion = success_count / success_history_length
            if success_proportion < population_protection:
                success_proportion = population_protection
            thresholds[i+1] = thresholds[i] + success_proportion
        # Scale down to make the max threshold = 1.
        thresholds *= (1.0 / thresholds[-1])
        return thresholds

    def basic_mutation(self, v0, v1, v2, f):
        '''
        Override basic_mutation to pick f_randomisation schemes from a
        roulette wheel.
        '''
        rand = numpy.random.rand()
        if rand < self.f_thresholds[1]:
            metadata = {'f_randomisation': 'static', 'f': f}
        elif rand < self.f_thresholds[2]:
            f = self.dither(f)
            metadata = {'f_randomisation': 'dither', 'f': f}
        else:
            f = self.jitter(f)
            metadata = {'f_randomisation': 'jitter', 'f': f}
        return v0 + (f * (v1 - v2)), metadata

    def roulette_mutation(self, i, f, r0):
        '''
        New function to pick a mutation scheme from a roulette wheel.
        '''
        rand = numpy.random.rand()
        if rand < self.mutation_scheme_thresholds[1]:
            mutant, metadata = self.de_rand_1(i, f, r0)
            metadata['mutation_scheme'] = 'de_rand_1'
        elif rand < self.mutation_scheme_thresholds[2]:
            mutant, metadata = self.de_current_to_best_1(i, f, r0)
            metadata['mutation_scheme'] = 'de_current_to_best_1'
        else:
            mutant, metadata = self.de_best_1(i, f, r0)
            metadata['mutation_scheme'] = 'de_best_1'
        return mutant, metadata

    def solution_commencing_hook(self):
        '''
        This hook is called after the population is initialised
        but before the solver starts. Could be used to define
        instance variables used in later hooks.
        '''
        self.c_randomisation = self.dither
        self.winning_c_vals = []
        self.winning_f_randomisers = []
        self.winning_mutation_schemes = []
        self.f_thresholds = numpy.array([0, 0.33333, 0.66667, 1])
        self.mutation_scheme_thresholds = numpy.array([0, 0.5, 0.75, 1])
        self.mutation_scheme = self.roulette_mutation

    def tournament_complete_hook(self, trial_cost, parent_cost, metadata):
        '''
        This hook is called after each tournament.
        It could be used to perform postprocessing or logical (i.e. self-
        adaptive) operations.
        '''
        if trial_cost < parent_cost:
            self.winning_c_vals.append(metadata['c'])
            self.winning_f_randomisers.append(metadata['f_randomisation'])
            self.winning_mutation_schemes.append(metadata['mutation_scheme'])

    def generation_complete_hook(self):
        '''
        This hook is called after each generation.
        It could be used to perform postprocessing or logical (i.e. self-
        adaptive) operations.
        '''
        update_period = 2
        if not self.generation % update_period and self.generation >= 4 and self.winning_c_vals:
            mean = numpy.mean(self.winning_c_vals)
            # Fix mean between 0 and 1
            mean = min(1, mean)
            mean = max(0, mean)
            self.c = mean
            printer('\nc has been adjusted to %s'%(str(self.c)))
            # Modify the roulette wheel for f_randomisers
            self.f_thresholds = self.roulette_wheel(
                    ['static', 'dither', 'jitter'], self.winning_f_randomisers)
            printer('f thresholds have been adjusted to %s'%(str(self.f_thresholds)))
            # And for mutation functions
            self.mutation_scheme_thresholds = self.roulette_wheel(
                    ['de_rand_1', 'de_current_to_best_1', 'de_best_1'], self.winning_mutation_schemes)
            printer('mutation scheme thresholds have been adjusted to %s'%(str(self.mutation_scheme_thresholds)))
            # Remove some system inertia
            self.winning_c_vals = []
            self.winning_f_randomisers = []
            self.winning_mutation_schemes = []

    def solution_complete_hook(self, victor):
        '''
        This hook is called after a solution is successfully found.
        It should return one or more values.
        '''
        return (victor, self.generation, self.function_evaluations,
                self.population_size, numpy.round(self.c,2), numpy.round(self.f_thresholds,2),
                numpy.round(self.mutation_scheme_thresholds,2))
