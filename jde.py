from deBase import DERand1Bin
import numpy
from random import choice


class jDE(DERand1Bin):
    """
    The original jDE by Brest et al., using one strategy
    (DE/rand/1/bin).
    """
    def __init__(self, *args, **kwargs):
        """
        Extend to encode random f and cr values onto each member of the population.
        """
        super(jDE, self).__init__(*args, **kwargs)
        for i in xrange(self.population.size):
            self.population.members[i].f = 0.1 + 0.9 * numpy.random.rand()
            self.population.members[i].cr = numpy.random.rand()

    def generateTrialMember(self, i):
        """
        Base f and cr upon parent member, or regenerate (p=0.1).
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
        # Perform the mutation and crossover operations
        mutant = self.mutation(i, f)
        trialMember = self.crossover(i, mutant, cr)
        # Note the parmeters used to generate the trial member & return
        trialMember.f = f
        trialMember.cr = cr
        return trialMember

    def selectNextGeneration(self, *args, **kwargs):
        """
        Update the 'master' f and cr with the mean values in the population.
        This is just for logging and could be safely removed from the algorithm.
        """
        super(jDE, self).selectNextGeneration(*args, **kwargs)
        self.f = numpy.mean([member.f for member in self.population.members])
        self.cr = numpy.mean([member.cr for member in self.population.members])
