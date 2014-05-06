from deBase import DERand1Bin
from jade import DECurrentToPBest1BinWithArchive, JADEWithArchive, JADE
from sade import SaDE
from jde import jDE
from mixins import LocalSearchMixin
import numpy


class HybridjDE(DECurrentToPBest1BinWithArchive, jDE):
    """
    jDE, hybridised to also inherit strategies, which may be rand/1/bin or
    Zhang and Sanderson's 'current-to-pbest/1/bin'
    """
    def __init__(self, *args, **kwargs):
        """
        Extend to encode strategies onto each member of the population.
        Start 50/50.
        """
        super(HybridjDE, self).__init__(*args, **kwargs)
        self.strategies = (DECurrentToPBest1BinWithArchive, DERand1Bin)
        for i in xrange(self.population.size):
            self.population.members[i].strategy = i%2

    def generateTrialMember(self, i):
        """
        Base f, cr and strategy upon parent member, or regenerate (p=0.1).
        """
        # Pick f and cr
        parent = self.population.members[i]
        if numpy.random.rand() > 0.1:
            f = parent.f
        else:
            f = 0.1 + 0.9 * numpy.random.rand()
        if numpy.random.rand() > 0.1:
            cr = parent.cr
        else:
            cr = numpy.random.rand()
        if numpy.random.rand() > 0.1:
            strategy = parent.strategy
        else:
            strategy = 1 if parent.strategy == 0 else 0
        # Perform the mutation and crossover operations
        mutant = self.strategies[strategy].mutation(self, i, f)
        trialMember = self.crossover(i, mutant, cr)
        # Note the parmeters used to generate the trial member & return
        trialMember.f = f
        trialMember.cr = cr
        trialMember.strategy = strategy
        return trialMember


class HybridJADE(JADEWithArchive, jDE):
    """
    Use the true 'self-adaptive' f change as seen in jDE instead of the
    Lehmer mean.
    """
    def generateTrialMember(self, i):
        parent = self.population.members[i]
        if numpy.random.rand() > 0.1:
            fi = parent.f
        else:
            fi = 0.1 + 0.9 * numpy.random.rand()
        mutant = self.mutation(i, fi, p=0.05)
        # cr is a normally distributed variable, truncated to [0, 1]
        cri = sorted((0, numpy.random.normal(self.cr, 0.1), 1))[1]
        trialMember = self.crossover(i, mutant, cri)
        # Mark the trial member with the parameters used to create it
        trialMember.f = fi
        trialMember.cr = cri
        return trialMember


class sadJADE(SaDE, JADE):
    """
    SaDE, but with JADE-style f and cr adaptation
    """
    def __init__(self, *args, **kwargs):
        """
        SaDE's (NOT JADEs) init, but with slightly modified strategy memories.
        """
        SaDE.__init__(self, *args, **kwargs)
        for i in xrange(len(self.strategies)):
            self.strategies[i]['crMemory'] = []
            self.strategies[i]['fMemory'] = []
            self.strategies[i]['f'] = 0.5

    def generateTrialMember(self, i):
        """
        Closely related to the JADE version
        """
        # Extract the strategy from the sample
        strategyIndex = self.sampledStrategies[i]
        strategy = self.strategies[strategyIndex]
        algorithm = strategy['algorithm']
        # f is Cauchy distributed variable, truncated to be 1 if fi > 1 or
        # regenerated if f <= 0
        while True:
            fi = min(strategy['f'] + 0.1 * numpy.random.standard_cauchy(), 1)
            if fi > 0:
                break
        mutant = algorithm.mutation(self, i, fi)
        # cr is a normally distributed variable, truncated to [0, 1]
        cri = sorted((0, numpy.random.normal(strategy['cr'], 0.1), 1))[1]
        trialMember = algorithm.crossover(self, i, mutant, cri)
        # Mark the trial member with the parameters used to create it
        trialMember.f = fi
        trialMember.cr = cri
        trialMember.strategy = strategyIndex
        return trialMember

    def generateTrialPopulation(self, *args, **kwargs):
        """
        Update strategy selection probabilities and create a strategy sample.
        Add a new row to the success and failure memories (old ones are
        deleted automatically by the deque).
        """
        # n = number of strategies in use. Called multiple times in this function.
        n = len(self.strategies)
        if self.generation > self.lp:
            self._updateStrategyProbabilities()
        self.sampledStrategies = self._stochasticUniversalSampleStrategies()
        # Augment all memories
        self.successMemory.append([0] * n)
        self.failureMemory.append([0] * n)
        return super(SaDE, self).generateTrialPopulation(*args, **kwargs)

    def trialMemberSuccess(self, i, trialMember):
        """
        Small amendment of the SaDE version.
        """
        self.strategies[trialMember.strategy]['crMemory'].append(trialMember.cr)
        self.strategies[trialMember.strategy]['fMemory'].append(trialMember.f)
        self.successMemory[-1][trialMember.strategy] += 1
        super(SaDE, self).trialMemberSuccess(i, trialMember)

    def selectNextGeneration(self, trialPopulation, c=0.1):
        """
        Override to include adaptive logic.
        c is an under-relaxation factor.
        Based on SaDE, with bits of modified JADE.
        """
        super(SaDE, self).selectNextGeneration(trialPopulation)
        # Update f and cr according to any successes. Note the use of the
        # Lehmer mean to give more weight to large f.
        for i in xrange(len(self.strategies)):
            s = self.strategies[i]
            if s['crMemory'] and s['fMemory']:
                self.strategies[i]['cr'] = (1 - c) * s['cr'] + c * numpy.mean(s['crMemory'])
                self.strategies[i]['crMemory'] = []
                try:
                    self.strategies[i]['f'] = (1 - c) * s['f'] + c * self._lehmerMean(s['fMemory'])
                except:
                    print s
                self.strategies[i]['fMemory'] = []


class LocalJADE(LocalSearchMixin, JADEWithArchive):
    """
    Implemented through multiple inheritance.
    """
    pass
