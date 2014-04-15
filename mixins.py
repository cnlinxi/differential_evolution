from multiprocessing import Pool, cpu_count
import csv
from collections import deque

"""
Mixins can be included with any DE algorithm in this suite to modify
or extend their functionality in some way.
"""


class ParallelCostMixin(object):
    """
    A mixin to allow the cost function to be evaluated on parallel CPUs.
    """
            
    def assignCosts(self, population):
        """
        Overridden to use the parallel map function
        """
        costs = self.pool.map(self.cost, population.vectors)
        self.functionEvaluations += population.size
        for i in xrange(population.size):
            population.members[i].cost = costs[i]
        return population
        
    def optimise(self, *args, **kwargs):
        """
        Extend the optimise function to start up and shut down a pool of workers.
        """
        cpus = cpu_count()
        # A pool will, with no arguments, contain 'cpu_count' workers.
        # The argument was included to be more explicit.
        self.pool = Pool(cpus)
        super(ParallelCostMixin, self).optimise(*args, **kwargs)
        self.pool.terminate()
        

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
            

class TwoDifferencesMixin(object):
    """
    Easily convert a single vector difference DE variant, e.g. DE/rand/1/bin,
    to a 2-vector difference variant, e.g. DE/rand/2/bin, by including this mix-in.
    """ 
    def mutation(self, *args, **kwargs):
        kwargs['n'] = 2
        return super(TwoDifferencesMixin, self).mutation(*args, **kwargs)