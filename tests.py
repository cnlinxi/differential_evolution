import unittest
import numpy as np
import copy
from population import Population, Member
import testFunctions


class TestPopulation(unittest.TestCase):
    """
    Tests for the population class.
    """
    def setUp(self):
        # Define a large population to enable accurate
        # checking of statistical parameters
        self.populationSize = 100
        self.dimensionality = 3
        boundary = np.array([1] * self.dimensionality)
        self.boundaryConstraints = (0 * boundary, boundary)
        self.population = Population(self.populationSize, self.boundaryConstraints)

    """ Tests for correct initialisation """
    def test_membersInitialised(self):
        self.assertTrue(all(isinstance(x, Member) for x in self.population.members))

    def test_membersWithinBoundary(self):
        for member in self.population.members:
            aboveLower = all(member.vector[i] > self.boundaryConstraints[0][i] for i in range(self.dimensionality))
            belowUpper = all(member.vector[i] < self.boundaryConstraints[1][i] for i in range(self.dimensionality))
        self.assertTrue(aboveLower and belowUpper)

    def test_memberCostsSetToInf(self):
        self.assertTrue(all(x.cost == np.inf for x in self.population.members))

    """ Tests for statistical properties """
    def test_sizeProperty(self):
        self.assertEqual(self.population.size, self.populationSize)

    def test_meanProperty(self):
        self.assertTrue(0.4 < np.mean(self.population.mean) < 0.6)

    def test_standardDeviationProperty(self):
        self.assertTrue(0.1 < np.mean(self.population.standardDeviation) < 0.4)

    def test_bestVectorIndex(self):
        rand = np.random.randint(self.populationSize)
        self.population.members[rand].cost = -50
        self.assertEqual(self.population.bestVectorIndex, rand)

    def test_worstVectorIndex(self):
        for i in range(self.populationSize):
            self.population.members[i].cost = 0
        rand = np.random.randint(self.populationSize)
        self.population.members[rand].cost = 50
        self.assertEqual(self.population.worstVectorIndex, rand)

    """ Tests for methods """
    def test_constrain(self):
        member = Member([10, 6, 3])
        min = [4, 4, 4]
        max = [7, 7, 7]
        member.constrain(min, max, sequential=True)
        self.assertTrue(np.array_equal(member.vector, [4, 6, 7]))


if __name__ == '__main__':
    unittest.main()
