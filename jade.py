import population
from deBase import DECurrentToPBest1Bin
from collections import deque
import numpy

class DECurrentToPBest1BinWithArchive(DECurrentToPBest1Bin):
    """
    Zhang and Sanderson describe how  past failures can be exploited as clues
    regarding promising solution directions using an 'archive'.
    """
    
    def __init__(self, *args, **kwargs):
        super(DECurrentToPBest1BinWithArchive, self).__init__(*args, **kwargs)
        # Archive is implemented here as a double-ended queue (deque).
        # As it overflows beyond np, the oldest items will be deleted.
        # The original JADE simply deleted items at random. The archive 
        # itself may also be tournament-selected.
        self.archive = deque(maxlen=self.population.size)
        
    def mutation(self, i, f, n=1, k=None, p=0.05):
        """
        A lot of this is, begrudgingly, copied and pasted from DECurrentToPBest1Bin;
        however, this allows for a form of inheritance in which this specific
        mutation method can be called from a child class without having to consider
        the side effects of other parents/mix-ins.
        """
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
        allMembers = self.population.members + list(self.archive)
        rMain = self._nmeri(n, self.population.size, exclude=[i, pBestIndex])
        rArchive = self._nmeri(n, len(allMembers), exclude=[i, pBestIndex] + rMain)
        while rMain:
            v1 = self.population.members[rMain.pop()].vector
            v2 = allMembers[rArchive.pop()].vector
            try:
                difference += v1 - v2
            except NameError:
                difference = v1 - v2
        return population.Member(baseVector + f * difference)
        
    def trialMemberSuccess(self, i, trialMember):
        """
        This function is extended to insert surpassed parents into the archive.
        """
        self.archive.append(self.population.members[i])
        super(DECurrentToPBest1BinWithArchive, self).trialMemberSuccess(i, trialMember)
        

class JADE(DECurrentToPBest1Bin):
    """
    An implementation of Zhang & Sanderson's adaptive JADE algorithm,
    without archive.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['f'] = 0.5
        kwargs['cr'] = 0.5
        super(JADE, self).__init__(*args, **kwargs)
        
    def _lehmerMean(self, a):
        """
        Returns the Lehmer mean of a list of numbers 'a'.
        """
        return sum(x*x for x in a) / float(sum(a))
        
    def generateTrialMember(self, i):
        """
        Override to include randomisation controls and attribute marking.
        """
        # f is Cauchy distributed variable, truncated to be 1 if fi > 1 or
        # regenerated if f <= 0
        while True:
            fi = min(self.f + 0.1 * numpy.random.standard_cauchy(), 1)
            if fi > 0:
                break
        mutant = self.mutation(i, fi, p=0.05)
        # cr is a normally distributed variable, truncated to [0, 1]
        cri = sorted((0, numpy.random.normal(self.cr, 0.1), 1))[1]
        trialMember = self.crossover(i, mutant, cri)
        # Mark the trial member with the parameters used to create it
        trialMember.f = fi
        trialMember.cr = cri
        return trialMember
        
    def trialMemberSuccess(self, i, trialMember):
        """
        This function is extended to log successful f and cr.
        """
        self.successfulCr.append(trialMember.cr)
        self.successfulF.append(trialMember.f)
        super(JADE, self).trialMemberSuccess(i, trialMember)
        
    def selectNextGeneration(self, trialPopulation, c=0.1):
        """
        Override to include adaptive logic.
        c is an under-relaxation factor.
        """
        self.successfulCr = []
        self.successfulF = []
        super(JADE, self).selectNextGeneration(trialPopulation)
        # Update f and cr according to any successes. Note the use of the
        # Lehmer mean to give more weight to large f.
        if self.successfulCr:
            self.cr = (1 - c) * self.cr + c * numpy.mean(self.successfulCr)
            self.f = (1 - c) * self.f + c * self._lehmerMean(self.successfulF)
            

class JADEWithArchive(DECurrentToPBest1BinWithArchive, JADE):
    """
    JADE with archive, implemented through multiple inheritance.
    """
    pass
    