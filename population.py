import numpy

"""
This file describes a population, as used in a
general class of evolutionary algorithm (EA).
"""


class Member(object):
    """
    A member of a population has a vector and an associated cost
    (initialised to infinity).
    Other attributes may be added as necessary.
    Also provides a 'constrain' method to brick-wall constrain
    the member's vector within a boundary.
    """
    def __init__(self, vector):
        self.cost = numpy.inf
        self.vector = numpy.asarray(vector)

    def constrain(self, minVector=None, maxVector=None, sequential=False, bind=True):
        """
        Adjust a member's vector to comply with bounding/sequential constraints
        """
        v = self.vector
        if bind:
            if minVector is not None:
                v = numpy.maximum(minVector, v)
            if maxVector is not None:
                v = numpy.minimum(maxVector, v)
        if sequential:
            v = numpy.sort(v)
        self.vector = v

    def __str__(self):
        """
        String representation of the member.
        """
        vector = numpy.around(self.vector, 2)
        cost = numpy.around(self.cost, 5)
        return '%s (cost %s)'%(vector, cost)


class Population(object):
    """
    A group of Members, associated with statistical parameters such as mean
    and standard deviation, and methods to randomly (re)generate the Members.
    """
    def __init__(self, size=None, boundaries=None, sequential=False, members=None):
        """
        This function creates a randomly-distributed initial population.
        Halton or Gaussian distributions could also be used. If the population
        is sequential, each vector will be sorted low-high.
        Alternatively, 'members' can be specified directly.
        """
        if members:
            self.members=members
        else:
            self.boundaries = boundaries
            self.sequential = sequential
            minVector, maxVector = self.boundaries
            assert len(minVector) == len(maxVector)
            minVector = numpy.asarray(minVector)
            maxVector = numpy.asarray(maxVector)
            mean = numpy.mean(numpy.column_stack((minVector, maxVector)), axis=1)
            range = maxVector - minVector
            # A blank container to hold the population whilst constructing
            self.members = []
            for i in xrange(size):
                # A random vector in the range -0.5 - 0.5
                vector = numpy.random.rand(len(minVector)) - 0.5
                # Manipulate it so that it meets the specified min/max conditions
                vector *= range
                vector += mean
                # Enforce sequential constraints if applicable
                if sequential:
                    vector = numpy.sort(vector)
                # Add the fully-constructed vector to the population
                self.members.append(Member(vector))

    @property
    def size(self):
        return len(self.members)

    @property
    def vectors(self):
        return [member.vector for member in self.members]

    @property
    def vectorArray(self):
        vectors = numpy.array(self.vectors)
        return numpy.column_stack(vectors)

    @property
    def costs(self):
        return [member.cost for member in self.members]

    @property
    def mean(self):
        return numpy.mean(self.vectorArray, axis=1)

    @property
    def standardDeviation(self):
        return numpy.std(self.vectorArray, axis=1)

    @property
    def bestVectorIndex(self):
        """
        Get the index of the best-performing member of the population
        """
        return min(xrange(len(self.costs)), key=self.costs.__getitem__)

    @property
    def worstVectorIndex(self):
        """
        Get the index of the best-performing member of the population
        """
        return max(xrange(len(self.costs)), key=self.costs.__getitem__)

    @property
    def bestVector(self):
        return self.members[self.bestVectorIndex]

    @property
    def worstVector(self):
        return self.members[self.worstVectorIndex]

    def __str__(self):
        return 'Population: size=%s, mean=%s, std=%s'%(self.size,
            self.mean, self.standardDeviation)
