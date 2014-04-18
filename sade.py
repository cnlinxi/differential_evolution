from deBase import DERand1Bin, DECurrentToPBest1Bin
from collections import deque
import itertools
import numpy
# Some of the following code can create numpy warnings.
# Tell numpy to raise exceptions instead so we can handle them.
numpy.seterr(all='raise')

"""
This file begins by deriving three of the four DE variants used by SaDE
(DE/rand/1/bin can be used directly.)

SaDE itself then inherits from these four variants.
Note that this is the 2009 update of SaDE, not the original 2005 version.
"""

class DECurrentToBest2Bin(DECurrentToPBest1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['p'] = 0
        kwargs['n'] = 2
        return DECurrentToPBest1Bin.mutation(self, *args, **kwargs)
        

class DERand2Bin(DERand1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['n'] = 2
        return DERand1Bin.mutation(self, *args, **kwargs)


class DECurrentToRand1(DECurrentToPBest1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['p'] = 1
        return DECurrentToPBest1Bin.mutation(self, *args, **kwargs)
        
    def crossover(self, parentIndex, mutant, cr):
        """
        This algorithm does not implement crossover to retain rotational invariance.
        """
        return mutant


class SaDE(DECurrentToBest2Bin, DERand2Bin, DECurrentToRand1, DERand1Bin):
    """
    An implementation of Qin et al.'s Self-adaptive Differential Evolution (SaDE).
    """
    
    def __init__(self, *args, **kwargs):
        """
        Upon initialisation, create a list of dicts containing
        information about SaDE's four strategies.
        """
        kwargs['f'] = 0.5
        kwargs['cr'] = 0.5
        super(SaDE, self).__init__(*args, **kwargs)
        self.lp = 50
        self.strategies = [
            {
                'algorithm': DERand1Bin,
                'probability': 0.25,
                'cr': 0.5, 
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToBest2Bin, 
                'probability': 0.25, 
                'cr': 0.5, 
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DERand2Bin,
                'probability': 0.25,
                'cr': 0.5,  
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToRand1,
                'probability': 0.25,
                'cr': 0.5,  
                'crMemory': deque(maxlen=self.lp)
            },
        ]
        # Note that deques delete old contents as they overflow beyond maxlen.
        self.successMemory = deque(maxlen=self.lp)
        self.failureMemory = deque(maxlen=self.lp)
        
    def _updateStrategyProbabilities(self):
        """
        Update the probability of each strategy being selected by examining the
        contents of the success and failure memories.
        """
        denominator = float(numpy.sum(self.successMemory) + numpy.sum(self.failureMemory))
        successes = numpy.sum(self.successMemory, axis=0)
        # 0.01 protects against null success rates
        unscaledProbabilities = [(s / denominator) + 0.01 for s in successes]
        # We want the probabilities scaled to 1.
        scalingFactor = 1 / sum(unscaledProbabilities)
        for i in xrange(len(self.strategies)):
            self.strategies[i]['probability'] = unscaledProbabilities[i] * scalingFactor
        
    def _stochasticUniversalSampleStrategies(self):
        """
        Returns a randomised list of NP strategy indices, sampled using the 
        Stochastic Universal Sampling technique, weighted by strategy probability.
        """
        # Assemble the sampling thresholds
        thresholds = []
        cumulativeProbability = 0
        for strategy in self.strategies:
            cumulativeProbability += strategy['probability']
            thresholds.append(cumulativeProbability)
        # Collect the sample
        sample = []
        interval = (1 / float(self.population.size))
        pointer =  interval * numpy.random.rand()
        while pointer < cumulativeProbability:
            for i, t in enumerate(thresholds):
                if pointer < t:
                    sample.append(i)
                    break
            pointer += interval
        numpy.random.shuffle(sample)
        return sample
        
        
    def _computeCrMedians(self):
        """
        Establish the median successful cr for each strategy in the last lp generations.
        """
        for i, strategy in enumerate(self.strategies):
            flattenedCr = list(itertools.chain.from_iterable(strategy['crMemory']))
            if flattenedCr:
                # Skip this step if there were no successes
                self.strategies[i]['cr'] = numpy.median(flattenedCr)
        
    def generateTrialMember(self, i):
        """
        Override to include randomisation controls and attribute marking.
        """
        # Extract the strategy from the sample
        strategyIndex = self.sampledStrategies[i]
        strategy = self.strategies[strategyIndex]
        algorithm = strategy['algorithm']
        # f is a normally distributed variable. SaDE does not truncate it.
        fi = numpy.random.normal(self.f, 0.3)
        # cr is a normally distributed variable, regenerated if outside [0, 1].
        while True:
            cri = numpy.random.normal(strategy['cr'], 0.1)
            if cri >= 0 and cri <= 1:
                break
        mutant = strategy['algorithm'].mutation(self, i, fi)
        trialMember = strategy['algorithm'].crossover(self, i, mutant, cri)
        # Mark the trial member with the parameters used to create it
        trialMember.strategy = strategyIndex
        trialMember.cr = cri
        return trialMember
        
    def generateTrialPopulation(self, *args, **kwargs):
        """
        Compute cr medians.
        Update strategy selection probabilities and create a strategy sample.
        Add a new row to the success, failure and cr memories (old ones are
        deleted automatically by the deque).
        """
        # n = number of strategies in use. Called multiple times in this method.
        n = len(self.strategies)
        if self.generation > self.lp:
            self._computeCrMedians()
            self._updateStrategyProbabilities()
        self.sampledStrategies = self._stochasticUniversalSampleStrategies()
        # Augment all memories
        self.successMemory.append([0] * n)
        self.failureMemory.append([0] * n)
        for i in xrange(n):
            self.strategies[i]['crMemory'].append([])  
        return super(SaDE, self).generateTrialPopulation(*args, **kwargs)
        
    def trialMemberSuccess(self, i, trialMember):
        """
        This function is extended to log successful cr and strategy.
        """
        self.strategies[trialMember.strategy]['crMemory'][-1].append(trialMember.cr)
        self.successMemory[-1][trialMember.strategy] += 1
        super(SaDE, self).trialMemberSuccess(i, trialMember)
        
    def trialMemberFailure(self, i, trialMember):
        """
        This function is extended to log unsuccessful strategies.
        """
        self.failureMemory[-1][trialMember.strategy] += 1
        super(SaDE, self).trialMemberFailure(i, trialMember)