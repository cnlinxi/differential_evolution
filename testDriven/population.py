"""
This file describes a population, as used in a 
general class of evolutionary algorithm (EA).
"""

import numpy as np

class Member(object):
    """
    A member of a population.
    """
    def __init__(self, vector):
        self.cost = np.inf
        self.vector = np.asarray(vector)
        
    def constrain(self, minVector, maxVector, bind=True, sequential=False):
        """
        Adjust a member's vector to comply with bounding/sequential constraints
        """
        vector = self.vector
        if bind:
            vector = numpy.maximum(minVector, vector)
            vector = numpy.minimum(maxVector, vector)
        if sequential:
            vector = np.sort(vector)
        self.vector = vector
        

class Population(object):
    """
    A group of Members, associated with statistical parameters such as
    mean and standard deviation, and methods to (re)generate the Members.
    """
    def __init__(self, size, boundaries, sequential=False):
        """
        This function creates a randomly-distributed initial population.
        Halton or Gaussian distributions could also be used. If the population
        is sequential, each vector will be sorted low-high.
        """
        self.boundaries = boundaries
        self.sequential = sequential
        minVector, maxVector = self.boundaries
        assert len(minVector) == len(maxVector)
        mean = np.mean(np.column_stack((minVector, maxVector)), axis=1)
        range = maxVector - minVector
        # A blank container to hold the population whilst constructing
        self.members = [] 
        for i in xrange(size):
            # A random vector in the range -0.5 - 0.5
            vector = np.random.rand(len(minVector)) - 0.5
            # Manipulate it so that it meets the specified min/max conditions
            vector *= range
            vector += mean
            # Enforce sequential constraints if applicable
            if sequential:
                vector = np.sort(vector)
            # Add the fully-constructed vector to the population
            self.members.append(Member(vector))
        
    @property
    def size(self):
        return len(self.members)
        
    @property
    def dimensionality(self):
        return len(self.members[0].vector)
        
    @property
    def vectors(self):
        return [member.vector for member in self.members]
        
    @property
    def vectorArray(self):
        vectors = np.array(self.vectors)
        return np.column_stack(vectors)
    
    @property  
    def costs(self):
        return [member.cost for member in self.members]
        
    @property
    def mean(self):
        return np.mean(self.vectorArray, axis=1)
        
    @property
    def standardDeviation(self):
        return np.std(self.vectorArray, axis=1)

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
    
    def reinitialise(self):
        """
        This method (re)initialises the population.
        """
        self.__init__(self.size, self.boundaries, self.sequential)
        
    def getVectorsByIndices(self, *args):
        return [self.members[i].vector for i in args]
        