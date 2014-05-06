import csv
import population
from multiprocessing import Pool, cpu_count

"""
Mixins can be included with any DE algorithm in this suite (unless otherwise
specified) to modify or extend their functionality in some way.
"""


class ParallelCostMixin(object):
    """
    A mixin to allow the cost function to be evaluated on parallel CPUs.
    """

    def computeCosts(self, vectors):
        """
        Overridden to use the parallel map function
        """
        return self.pool.map(self.cost, vectors)

    def optimise(self, *args, **kwargs):
        """
        Extend the optimise function to start up and shut down a pool of workers.
        """
        cpus = cpu_count()
        # A pool will, with no arguments, contain 'cpu_count' workers.
        # The argument was included to be more explicit.
        self.pool = Pool(cpus)
        bestVector = super(ParallelCostMixin, self).optimise(*args, **kwargs)
        self.pool.terminate()
        return bestVector


class LoggingMixin(object):
    """
    Exports key solution data to CSV.
    Accepts a UUID parameter to track the CSV file.
    """

    def __init__(self, *args, **kwargs):
        self.csvFilename = '%s.csv'%(kwargs.pop('uuid'))
        super(LoggingMixin, self).__init__(*args, **kwargs)

    def selectNextGeneration(self, *args, **kwargs):
        """
        Extend selectNextGeneration (the last function called in each generation)
        to log key solution data from that generation.
        """
        super(LoggingMixin, self).selectNextGeneration(*args, **kwargs)
        with open(self.csvFilename, 'ab') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.functionEvaluations,
                self.population.bestVector.cost, self.cr, self.f])


class ValueToReachMixin(object):
    """
    Adds an extra option, to be specified upon instantiation, to terminate
    the optimisation process once a certain cost function value has been reached.
    This works in addition to (rather than instead of) other termination criteria.
    """

    def __init__(self, *args, **kwargs):
        self.valueToReach = kwargs.pop('valueToReach')
        super(ValueToReachMixin, self).__init__(*args, **kwargs)

    def terminationCriterion(self):
        return (super(ValueToReachMixin, self).terminationCriterion() or
            self.population.bestVector.cost < self.valueToReach)


class LocalSearchMixin(object):
    """
    Adds a basic local search to DE. This will, at present, only work for algorithms
    which do not encode additional information on population members like f, cr etc.
    It seems to be promising at low, but not high, dimensionality.
    """

    def generateTrialPopulation(self, *args, **kwargs):
        """
        Add the mean vector to end of the trial population.
        """
        trialPop = super(LocalSearchMixin, self).generateTrialPopulation(*args, **kwargs)
        trialPop.members.append(population.Member(self.population.mean))
        return trialPop

    def selectNextGeneration(self, trialPopulation):
        """
        Replace the worst vector with the mean,
        if the mean is better than the median cost.
        """
        # Remove the last member from the trial population (which we know is the mean).
        meanMember = trialPopulation.members.pop()
        # Note the use of integer (floor) division.
        medianMember = self.population.members[self.population.size / 2]
        if meanMember.cost < medianMember.cost:
            # Note that the population is ordered by cost, low-high.
            self.population.members[-1] = meanMember
        super(LocalSearchMixin, self).selectNextGeneration(trialPopulation)
