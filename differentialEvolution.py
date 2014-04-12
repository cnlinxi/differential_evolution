# We rely on np for array/vector operations and advanced maths.
import numpy as np
import time, datetime
import sys
import os
from multiprocessing import Pool, cpu_count
import population
import csv

class NotConvergedException(Exception):
    """
    Exception raised when a solution is not found
    """
    pass
    
    
def printer(text, output='stdout'):
    """
    A general form of the print function, to change the output
    location in one line.
    """
    if output == 'stdout':
        print text
    elif output == 'stderr':
        sys.__stderr__.write(str(text) + os.linesep)
        sys.__stderr__.flush()
        
LOW_CR = 0.1
HIGH_CR = 0.9

class DifferentialEvolution(object):
    """
    A class-based approach to the Differential Evolution (DE) problem.

    DE is an optimisation algorithm that minimises an n-dimensional
    cost function using one or more mutation functions.

    This is a basic version, as set out by Storn and Price in
    'Differential Evolution - A Practical Approach to Global Optimisation'.
    """
    
    def __init__(self, costFunction, bounds, populationSize=None, f=None, cr=None, 
        mutator='de/rand/1/bin', baseVectorSelector='permuted', fRandomiser='randomInterval', 
        crRandomiser='static', maxFunctionEvals=10000, convergenceFunction='std',
        convergenceStd=0.01, vtr=None, absoluteBounds=False, sequential=False, 
        verbosity=1, huddling=False, multiprocessing=False, outputLoc='stderr', 
        learningPeriod=5):
        """
        Specify the problem parameters and initialise the population.
        """
        # Start by listing our (discrete) choices
        self.mutators = {
            'de/rand/1/bin': self.deRand1,
            'de/best/1/bin': self.deBest1,
            'de/current_to_best/1/bin': self.deCurrentToBest1,
            'de/rand/2/bin': self.deRand2,
            'de/best/2/bin': self.deBest2,
            'de/rand_then_best/1/bin': self.deRandThenBest1,
        }
        self.baseVectorSelectors = {
            'random': self.randomBaseVectorSelection,
            'permuted': self.permutedBaseVectorSelection,
            'offset': self.randomOffsetBaseVectorSelection,
        }
        self.convergenceFunctions = {
            'std': self.stdConvergence,
            'vtr': self.vtrConvergence,
        }
        self.fRandomisers = {
            'static': lambda f: f,  # Just return f when this is called.
            'dither': self.dither,
            'jitter': self.jitter,
            'randomInterval': self.fInRandomInterval
        }
        self.crRandomisers = {
            'static': lambda cr: cr,  # Just return c when this is called.
            'dither': self.dither,
        }
        # Declare constructor arguments as instance variables
        self.cost = costFunction
        self.minVector = np.asarray(bounds[0])
        self.maxVector = np.asarray(bounds[1])
        self.dimensionality = len(self.minVector)
        # Check for 'phantom' dimensions that are constrained to a single value.
        self.phantomIndices = []
        for i, x in enumerate(self.maxVector - self.minVector):
            if x==0:
                self.phantomIndices.append(i)
        # Default value for population size is 5 times the
        # problem dimensionality (2D = 10, 3D = 15, etc.).
        if not populationSize:
            populationSize = 10 * (self.dimensionality - len(self.phantomIndices))
            self.populationSize = populationSize#min(populationSize, 40)
        else:
            self.populationSize = populationSize
        # Selection of base vector scheme
        self.baseVectorSelector = baseVectorSelector
        self.mutator = mutator
        self.convergenceFunction = convergenceFunction
        # Mutation scaling factor. Recommended 0.5 < f < 1.2
        if f==None and fRandomiser != 'randomInterval':
            self.f = 0.9
        else:
            self.f = f
        if f and fRandomiser=='randomInterval':
            self.fRandomiser = 'dither'
        self.fRandomiser = fRandomiser
        # Crossover factor (see def crossover)
        self.cr = cr
        # Select c distribution
        if cr:
            self.crRandomiser = crRandomiser
        else:
            self.crRandomiser = self.selfAdaptiveCr
            self.highCrProbability = 0.5
        self.learningPeriod = learningPeriod
        # Number of generations before the program terminates regardless
        # of convergence
        self.maxFunctionEvals = maxFunctionEvals
        # Maximum standard deviation of the population for the solution to be
        # considered converged. Only used if convergenceFunction = stdConvergence
        self.convergenceStd = convergenceStd
        # Value to reach. Only used if convergenceFunction = vtrConvergence
        self.valueToReach = vtr
        # Are the boundaries of the problem absolute, i.e. are mutations outside
        # the bounding vectors banned?
        self.absoluteBounds = absoluteBounds
        # Turn 'huddling' (see def huddle) on or off. Default = off.
        self.huddling = huddling
        # Select logging amount. 0=silent, 1=basic, 2=verbose
        self.verbosity = verbosity
        self.multiprocessing = multiprocessing
        self.sequential = sequential
        # Initialise the population
        self.population = population.Population(
            self.populationSize, (self.minVector, self.maxVector), self.sequential)
        
        
    def _stringToFunctionIfString(self, attr, dictionary):
        """
        Utility to convert a string to a function by lookup in a
        dictionary, or leave the function alone if it is already a function,
        then set self.attr equal to that function.
        """
        funcOrString = getattr(self, attr)
        if isinstance(funcOrString, basestring):
            func = dictionary[funcOrString]
        else:
            func = funcOrString
        setattr(self, attr, func)
        
    # The following functions are used during mutation to
    # select the base vector, v0.
    def randomBaseVectorSelection(self):
        """
        Base vectors are selected with a random and independent probability.
        The only condition is that randoms[i] != i
        """
        randoms = [0] # Arbitrary initialisation to imitate a do-while loop
        while any(randoms[i] == i for i in randoms):
            randoms = np.random.randint(self.populationSize, size=self.populationSize)
        return randoms

    def permutedBaseVectorSelection(self):
        """
        Base vectors are selected randomly but with dependent probability.
        Each vector is used as v0 only once per generation.
        """
        randoms = np.random.permutation(self.populationSize)
        while any(randoms[i] == i for i in randoms):
            np.random.shuffle(randoms)
        return randoms

    def randomOffsetBaseVectorSelection(self):
        """
        Base vectors are selected at a random but unchanging offset from the
        parent vector.
        """
        randomOffset = np.random.randint(self.populationSize)
        return [(i + randomOffset) % self.populationSize for i in xrange(self.populationSize)]
        
    def dither(self, x, sigma=0.2):
        """
        Returns a scalar based on a normal distribution about x
        with a standard deviation of sigma.
        """
        return np.random.normal(x, sigma)

    def jitter(self, x, sigma=0.2):
        """
        Returns a vector based on a normal distribution about x
        with a standard deviation of sigma.
        """
        return np.random.normal(x, sigma, self.population.dimensionality)
        
    def fInRandomInterval(self, x):
        """
        Returns a random scalar between 0.5 and 1.
        """
        return 0.5 + np.random.rand()/2.0 
        
    def selfAdaptiveCr(self, cr):
        """
        Choose cr from a roulette wheel
        """
        rand = np.random.rand()
        if rand > self.highCrProbability:
            return LOW_CR
        else:
            return HIGH_CR
        
    # The following functions concern mutation operations.
    # They are self-adaptive ready, but there is no self-adaptive logic here.
    def _nmeri(self, n, maximum, minimum=0, notEqualTo=[]):
        """
        Helper function to return N Mutually Exclusive Random Integers (nmeri)
        in the range [minimum, maximum). Optionally takes a 'not_equal_to'
        argument; a list of integers which will be excluded from the set.
        """
        selected = set()
        while len(selected) < n:
            rand = np.random.randint(minimum, maximum)
            # No need to check if rand in selected as selected is a set.
            if rand not in notEqualTo:
                selected.add(rand)
        return tuple(selected)

    def basicMutation(self, v0, v1, v2, f):
        """
        Mutation helper function, called by all mutation types.
        Returns v0 + [f * (v1 - v2)], where v0, v1 and v2 are vectors.
        """
        return v0 + (f * (v1 - v2))

    def deRand1(self, i, f, r0):
        """
        'Classic' DE mutation - combine three random vectors.
        """
        r1, r2 = self._nmeri(2, self.populationSize, notEqualTo=[i, r0])
        v0, v1, v2 = self.population.getVectorsByIndices(r0, r1, r2)
        return self.basicMutation(v0, v1, v2, f)

    def deBest1(self, i, f, r0):
        """
        Variation on classic DE, using the best-so-far vector as v0.
        r0 is allowed as an argument for consistency, but is not used.
        """
        rBest = self.population.bestVectorIndex
        r1, r2 = self._nmeri(2, self.populationSize, notEqualTo=[i, rBest])
        v0, v1, v2 = self.population.getVectorsByIndices(rBest, r1, r2)
        return self.basicMutation(v0, v1, v2, f)

    def deCurrentToBest1(self, i, f, r0):
        """
        Hybrid of de/rand/1 and de/best/1. r0 is again ignored.
        """
        rBest = self.population.bestVectorIndex
        vi, vBest = self.population.getVectorsByIndices(i, rBest)
        currentToBest = self.basicMutation(vi, vBest, vi, f)
        r1, r2 = self._nmeri(2, self.populationSize, notEqualTo=[i, rBest])
        v1, v2 = self.population.getVectorsByIndices(r1, r2)
        return self.basicMutation(currentToBest, v1, v2, f)

    def deRand2(self, i, f, r0):
        """
        Like de/rand/1, but adds two random scaled vectors.
        Modified from Qin and Suganthan by using f/2.
        """
        r1, r2, r3, r4 = self._nmeri(4, self.populationSize, notEqualTo=[i, r0])
        v0, v1, v2 = self.population.getVectorsByIndices(r0, r1, r2)
        mutant = self.basicMutation(v0, v1, v2, f*0.5)
        v3, v4 = self.population.getVectorsByIndices(r3, r4)
        return self.basicMutation(mutant, v3, v4, f*0.5)

    def deBest2(self, i, f, r0):
        """
        Like de/best/1, but adds two random scaled vectors.
        Modified from Qin and Suganthan by using f/2.
        """
        rBest = self.population.bestVectorIndex
        r1, r2, r3, r4 = self._nmeri(4, self.populationSize, notEqualTo=[i, r0])
        v0, v1, v2 = self.population.getVectorsByIndices(rBest, r1, r2)
        mutant = self.basicMutation(v0, v1, v2, f*0.5)
        v3, v4 = self.population.getVectorsByIndices(r3, r4)
        return self.basicMutation(mutant, v3, v4, f*0.5)

    def deRandThenBest1(self, i, f, r0):
        """
        If the generation is odd, use de/rand/1.
        If the generation is even, use de/best/1.
        """
        if self.generation % 2:
            return self.deRand1(i, f, r0)
        else:
            return self.deBest1(i, f, r0)
        
    def mutation(self, i, baseVectorIndex):
        """
        Mutate a vector in the population.
        """
        f = self.fRandomiser(self.f)
        vector = self.mutator(i, f, baseVectorIndex)
        mutant = population.Member(vector)
        mutant.f = f
        return mutant

    def crossover(self, v1, mutantMember):
        """
        Creates a trial vector by crossing v1 with v2 to create v3.
        The probability of a v2 element being selected over a v1 element is c,
        the crossover factor. There also exists an 'iRand' to guarantee that
        at least one mutant value is chosen in the crossover.
        """
        cr = self.crRandomiser(self.cr)
        i_rand = np.random.randint(self.dimensionality)
        while i_rand in self.phantomIndices:
            i_rand = np.random.randint(self.dimensionality)
        random_array = np.random.rand(self.dimensionality)
        for i, random_number in enumerate(random_array):
            if random_number > cr and i != i_rand:
                mutantMember.vector[i] = v1[i]
        mutantMember.cr = cr
        return mutantMember
        
    def generateTrialPopulation(self):
        """
        This function creates a trial population from an existing population
        using mutation and crossover operations.
        It returns a trial population.
        """
        trialMembers = []
        baseVectorIndices = self.baseVectorSelector()
        for i, baseVectorIndex in enumerate(baseVectorIndices):
            trialMember = self.mutation(i, baseVectorIndex)
            trialMember = self.crossover(self.population.members[i].vector, trialMember)
            trialMember.constrain(self.minVector, self.maxVector, 
                self.sequential, self.absoluteBounds)
            trialMembers.append(trialMember)
        return trialMembers
            
    def updatePopulation(self, trialPopulation, costs):
        """
        Replace vectors in the population whose costs are higher than the trial vectors.
        """
        if self.huddling:
            huddleMember = trialPopulation.pop(-1)
            huddleMember.cost = costs.pop(-1)
            wvi = self.population.worstVectorIndex
            if huddleMember.cost < self.population.members[wvi].cost:
                self.population.members[wvi] = huddleMember
        for i, trialMember in enumerate(trialPopulation):
            trialMember.cost = costs[i]
            # Less than or equal is important to avoid stagnation
            # in quantised landscapes.
            if trialMember.cost <= self.population.members[i].cost:
                self.population.members[i] = trialMember
                self.successfulCr.append(trialMember.cr)
                
    # One of these functions is called at the end of each generation. The
    # convergence functions given here are selectable termination criteria.
    def stdConvergence(self):
        """
        Returns True if the standard deviation of the population is below the
        specified value in all dimensions.
        """
        return min(self.population.standardDeviation) < self.convergenceStd

    def vtrConvergence(self):
        """
        Returns True when the lowest function cost dips below a given value.
        (a Value To Reach)
        """
        return min(self.population.costs) < self.valueToReach

    # The following functions log progress. Whether one or all of them is called
    # depends on the verbosity setting.
    def basicLogging(self):
        """
        This basic logging information is always shown unless verbosity is
        set to 0 (silent).
        """
        currentTime = datetime.datetime.now()
        timeString = currentTime.strftime("%A, %d %B, %Y %I:%M%p")
        printer('\n2014 Blake Hemingway', outputLoc)
        printer('The University of Sheffield', outputLoc)
        printer('\nRun started on %s'%(timeString), outputLoc)
        printer('\nInitial solution parameters:\n', outputLoc)
        printer('Scaling factor:  \t%s'%(self.f), outputLoc)
        printer('Crossover factor:\t%s'%(self.cr), outputLoc)
        printer('Population size: \t%s\n'%(self.populationSize), outputLoc)

    def logSolution(self, generation):
        """
        Log data about the progress of the solution such as mean,
        standard deviation and current leader. Only called if the
        'verbose_logging' parameter is set
        """
        best = np.round(min(self.population.costs), 3)
        printer('\nAt generation %s:'%(generation), outputLoc)
        printer('Mean values: %s'%(str(self.population.mean)), outputLoc)
        printer('std values: %s'%(str(self.population.standardDeviation)), outputLoc)
        printer('Best-so-far function value: %s\n'%(best), outputLoc)

    def optimise(self):
        """
        The main method. Call this method to run the optimisation.
        """
        # Get functions corresponding to strings by using them as keys in lookup
        self._stringToFunctionIfString('baseVectorSelector', self.baseVectorSelectors)
        self._stringToFunctionIfString('mutator', self.mutators)
        self._stringToFunctionIfString('convergenceFunction', self.convergenceFunctions)
        self._stringToFunctionIfString('fRandomiser', self.fRandomisers)
        self._stringToFunctionIfString('crRandomiser', self.crRandomisers)
        # Initialise the logging process
        if self.verbosity > 0:
            self.basicLogging()
        # Redefine map as a parallel function if we are multiprocessing.
        if self.multiprocessing:
            cpus = cpu_count()
            pool = Pool(cpus)
            mapper = pool.map
        else:
            pool = None
            mapper = map
        costs = mapper(self.cost, self.population.vectors)
        self.functionEvaluations = self.populationSize
        self.successfulCr = []
        self.generation = 0
        while self.functionEvaluations < self.maxFunctionEvals:
            self.generation += 1
            # If logging, show output
            if self.verbosity > 1:
                self.logSolution(self.generation)
            # Evolve (mutate/crossover) the next generation
            trialPopulation = self.generateTrialPopulation()
            if self.huddling:
                trialPopulation.append(population.Member(self.population.mean))
            # Evaluate the next generation
            costs = mapper(self.cost, [member.vector for member in trialPopulation])
            self.functionEvaluations += len(trialPopulation)
            # Insert improvements
            self.updatePopulation(trialPopulation, costs)
            # Check for solution convergence
            convergence = self.convergenceFunction()
            if convergence:
                # Return the final population, with some metadata.
                try:
                    pool.terminate()
                except:
                    pass
                return self.population, self.functionEvaluations, self.generation
            # Otherwise, run some self-adaptive logic
            elif self.crRandomiser == self.selfAdaptiveCr and self.generation % self.learningPeriod == 0:
                if self.successfulCr:
                    self.cr = np.mean(self.successfulCr)
                    self.highCrProbability = sorted([0.05, self.successfulCr.count(HIGH_CR) / float(len(self.successfulCr)), 0.95])[1]
                self.successfulCr = []
        # If we get to here, we haven't achieved convergence. Raise an error.
        try:
            pool.terminate()
        except:
            pass
        raise NotConvergedException('The solution did not converge', self.population.bestVector.cost)
        
        
class DifferentialEvolution2(DifferentialEvolution):
    """
    As above, but with a modular cost function API,
    (i.e. cost is a file).
    If the keyword argument 'dimensions' is passed,
    the code will look for a 'getBounds' method which
    accepts a dimensions parameter. Otherwise, the
    code will look for a 'bounds' attribute. 
    Similarly, a 'vtr' attribute or 'getVtr(d)'
    function may be used to specify the value to
    reach.
    """
    def __init__(self, costModule, **kwargs):
        kwargs['costFunction'] = costModule.cost
        if 'dimensions' in kwargs:
            d = kwargs.pop('dimensions')
            kwargs['bounds'] = costModule.getBounds(d)
        else:
            kwargs['bounds'] = costModule.bounds
        if hasattr(costModule, 'vtr'):
            kwargs['vtr'] = costModule.vtr
            kwargs['convergenceFunction'] = 'vtr'
        elif 'dimensions' in kwargs:
            try:
                kwargs['vtr'] = costModule.getVtr(d)
                kwargs['convergenceFunction'] = 'vtr'
            except AttributeError:
                kwargs['convergenceFunction'] = 'std'
        else:
            kwargs['convergenceFunction'] = 'std'
        try:
            kwargs['convergenceStd'] = costModule.convergenceStd
        except AttributeError:
            pass
        try:
            kwargs['absoluteBounds'] = costModule.absoluteBounds
        except AttributeError:
            pass
        try:
            kwargs['sequential'] = costModule.sequential
        except AttributeError:
            pass
        super(DifferentialEvolution2, self).__init__(**kwargs)
        
        
    
class LoggingDifferentialEvolution(DifferentialEvolution2):
    """
    As above, but exports key solution data to CSV as it works.
    Accepts a UUID parameter to track the CSV file.
    """
    def __init__(self, costModule, uuid, **kwargs):
        super(LoggingDifferentialEvolution, self).__init__(costModule, **kwargs)
        self.csvFilename = '%s.csv'%(uuid)
        
    def updatePopulation(self, trialPopulation, costs):
        """
        Replace vectors in the population whose costs are higher than the trial vectors.
        """
        super(LoggingDifferentialEvolution, self).updatePopulation(trialPopulation, costs)
        if self.cr:
            logCr = self.cr
        else:
            logCr = 0.5
        if self.f:
            logF = self.f
        else:
            logF = 0.75
        with open(self.csvFilename, 'ab') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.functionEvaluations,
                self.population.bestVector.cost, logCr, logF])

        