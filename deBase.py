# We rely on numpy for array/vector operations and advanced maths.
import numpy
import population

"""
Blake Hemingway, University of Sheffield, 2014

Differential Evolution (DE) is an optimisation algorithm that minimises 
an n-dimensional cost function using a population-based approach.

This file contains base classes representing generalised forms of the
most common DE variants.

More advanced forms of DE, with, for example, adaptive parameter controls, 
may be implemented as subclasses of these.
Options (e.g. parallel cost function evaluation, logging to CSV file, different
termination criteria, crossover techniques etc.) can be implemented as mix-ins.
"""


class DERand1Bin(object):
    """
    This is 'classic' DE as outlined in Storn and Price's "Differential Evolution:
    A Practical Approach to Global Optimization".
    """
    
    def __init__(self, costFile, np=50, f=0.8, cr=0.9, maxFunctionEvals=50000):
        """
        This function is called when the DifferentialEvolution class is instantiated.
        A 'costFile' must be passed: this should be an included Python module
        containing the following methods:
        
        - cost(x): returning a scalar when passed a single vector argument x
        - getBounds: returning a tuple, length 2, of the initialisation region.
        
        A Boolean 'absoluteBounds' may also be set. If this is True, mutations outside
        the initialisation region will be banned.
        
        Control parameters np (population size), f (mutation scaling factor) and
        cr (crossover ratio) can be specified, or left as the 'standard' values
        as stated by Qin & Suganthan (np) and Storn (cr, f).
        np will be, by default, conservatively large for low-dimensional problems.
        np = 10d, where d is problem dimensionality, is recommended in such cases.
        """
        # Get cost function
        self.cost = costFile.cost
        # Get problem boundaries
        self.minVector, self.maxVector = costFile.getBounds()
        # Infer the problem dimensionality from one of these (arbitrarily)
        self.d = len(self.minVector)
        # Initialise population randomly within the boundaries.
        self.population = population.Population(
            size=np, boundaries=(self.minVector, self.maxVector))
        # Are mutations outside these boundaries banned?
        try:
            self.absoluteBounds = costFile.absoluteBounds
        except AttributeError:
            self.absoluteBounds = False
        #print self.absoluteBounds
        # Number of function evaluations before the program terminates 
        self.maxFunctionEvals = maxFunctionEvals
        # The number of function evaluations now is, obviously, 0.
        self.functionEvaluations = 0
        # Define solution parameters
        self.cr = cr
        self.f = f
        
    def _nmeri(self, n, maximum, exclude=[]):
        """
        Helper function to return N Mutually Exclusive Random Integers (nmeri)
        in the range [0, maximum). Optionally takes an 'exclude'
        argument; a list of integers which will be excluded from the set.
        """
        selected = set()
        while len(selected) < n:
            rand = numpy.random.randint(maximum)
            # No need to check if rand in selected as selected is a set.
            if rand not in exclude:
                selected.add(rand)
        return list(selected)
        
    def mutation(self, i, f, n=1):
        """
        The mutation style used by de/rand/n.
        Create a mutant individual by adding n scaled vector differences
        to a base vector in the main population.
        """
        r = self._nmeri(1 + 2*n, self.population.size, exclude=[i])
        baseVector = self.population.members[r.pop()].vector
        while r:
            v1, v2 = [self.population.members[j].vector for j in (r.pop(), r.pop())]
            try:
                difference += v1 - v2
            except NameError:
                difference = v1 - v2
        return population.Member(baseVector + f * difference)
        
    def crossover(self, parentIndex, mutant, cr):
        """
        Create a trial member by crossing a parent with a mutant.
        This function uses a binomial distribution to do so.
        The probability of a mutant element being selected over a parent element is 
        cr, the crossover factor. There also exists an 'iRand' to guarantee that
        at least one mutant value is chosen.
        """
        parent = self.population.members[parentIndex]
        iRand = numpy.random.randint(self.d)
        for i in xrange(self.d):
            if numpy.random.rand() > cr and i != iRand:
                mutant.vector[i] = parent.vector[i]
        # 'mutant' is now not strictly a mutant but a trial member.
        return mutant
        
    def generateTrialMember(self, i):
        """
        Generate a single trial member by calling mutation and crossover operations.
        """
        mutant = self.mutation(i, self.f)
        trialMember = self.crossover(i, mutant, self.cr)
        return trialMember
        
    def generateTrialPopulation(self, np):
        """
        Create a trial population (size np) from an existing population.
        Return a population object.
        """
        trialMembers = []
        for i in xrange(np):
            trialMember = self.generateTrialMember(i)
            if self.absoluteBounds:
                trialMember.constrain(self.minVector, self.maxVector)
            trialMembers.append(trialMember)
        return population.Population(members=trialMembers)
        
    def assignCosts(self, population):
        """
        Compute and assign cost function values to each member of the passed 
        population object by considering the member vectors. 
        Return the modified population.
        """
        costs = map(self.cost, population.vectors)
        self.functionEvaluations += population.size
        for i in xrange(population.size):
            population.members[i].cost = costs[i]
        return population
        
    def trialMemberSuccess(self, i, trialMember):
        """
        This function is called in the event of trialMember being found to be 
        superior to its parent with index i in the population.
        """
        self.population.members[i] = trialMember
        
    def trialMemberFailure(self, i, trialMember):
        """
        This function is called in the event of trialMember being found to be 
        inferior to its parent with index i in the population.
        """
        pass
        
    def selectNextGeneration(self, trialPopulation):
        """
        Compare the main population with the trial population by cost.
        """
        for i, trialMember in enumerate(trialPopulation.members):
            # <= (not <) is important to avoid stagnation in quantised landscapes.
            if trialMember.cost <= self.population.members[i].cost:
                self.trialMemberSuccess(i, trialMember)
            else:
                self.trialMemberFailure(i, trialMember)
                
    def terminationCriterion(self):
        """
        Termination is based on a limited number of function evaluations.
        """
        return self.functionEvaluations >= self.maxFunctionEvals
        
    def optimise(self):
        """
        The main method. Call this method to run the optimisation.
        """
        self.population = self.assignCosts(self.population)
        self.generation = 0
        while self.terminationCriterion() == False:
            self.generation += 1
            # Generate (mutate/crossover) a trial population
            trialPopulation = self.generateTrialPopulation(self.population.size)
            # Evaluate the trial population
            trialPopulation = self.assignCosts(trialPopulation)
            # Insert improvements
            self.selectNextGeneration(trialPopulation)
        return self.population.bestVector
        

class DECurrentToPBest1Bin(DERand1Bin):
    """
    Constructs mutants in accordance with the following procedure:
    
    u_i = x_i + k*(x_pbest - x_i) + f_i*(x_r1 - x_r2)
    
    Where x_pbest is randomly chosen as one of the top 100 p% individuals,
    
    The following algorithms are specific cases:
    
    - DE/best/1/bin (k = 1, p = 0)
    - DE/current-to-best/1/bin (k = fi, p = 0)
    - DE/current-to-rand/1/bin (k = fi, p = 1)
    
    k is taken as equal to fi unless otherwise specified in the constructor.
    k may also be callable (e.g. a random number generating function),
    in which case it will be evaluated without arguments.
    p = 0.05 by default, as given in Zhang and Sanderson's JADE.
    """
        
    def mutation(self, i, f, n=1, k=None, p=0.05):
        if k is None:
            k = f
        else:
            try:
                k = k()
            except TypeError:
                pass
        # Obtain one of the p-best vectors at random (population is sorted by cost)
        if p:
            maxPBestIndex = numpy.ceil(p * self.population.size)
            pBestIndex = numpy.random.randint(maxPBestIndex)
        else:
            pBestIndex = 0
        pBestVector = self.population.members[pBestIndex].vector
        # Obtain the current (aka target) vector
        currentVector = self.population.members[i].vector
        # Compute the base vector
        baseVector = currentVector + k * (pBestVector - currentVector)
        # Compute a difference vector
        r = self._nmeri(2*n, self.population.size, exclude=[i, pBestIndex])
        while r:
            v1, v2 = [self.population.members[j].vector for j in (r.pop(), r.pop())]
            try:
                difference += v1 - v2
            except NameError:
                difference = v1 - v2
        return population.Member(baseVector + f * difference)
        
    def generateTrialPopulation(self, *args, **kwargs):
        """
        Sort main population by cost before generating trial population.
        """
        self.population.members.sort(key=lambda x: x.cost)
        return super(DECurrentToPBest1Bin, self).generateTrialPopulation(*args, **kwargs)