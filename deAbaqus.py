from abaqus import *
from abaqusConstants import *
from odbAccess import *
from jade import JADEWithArchive
from mixins import ParallelCostMixin
from scipy import spatial
import numpy
import os
import time

class AbaqusNodeKDTree(spatial.KDTree):
    '''
    Operates on a list of Abaqus MeshNode objects.
    'query' method returns the instance name and label of a node
    closest to a co-ordinate point (x, y, z).
    '''
    def __init__(self, nodeObjects, *args, **kwargs):
        # Split the objects out into two parallel lists: 
        self.labels = [(node.instanceName, node.label) for node in nodeObjects]
        data = [node.coordinates for node in nodeObjects]
        super(AbaqusNodeKDTree, self).__init__(data, *args, **kwargs)
        
    def query(self, x, *args, **kwargs):
        distance, dataIndex = super(AbaqusNodeKDTree, self).query(x, *args, **kwargs)
        return self.labels[dataIndex]
        

class AbaqusJADE(ParallelCostMixin, JADEWithArchive):
    """
    Extensions to JADE for use in Abaqus.
    """
    def __init__(self, *args, **kwargs):
        """
        We need more from the cost file, i.e. a model and constraints.
        From this model, create a KDTree instance.
        """
        super(AbaqusJADE, self).__init__(*args, **kwargs)
        self.model = costFile.getModel()
        self.constraints = costFile.getConstraints()
        # Dict to hold the node combinations visited, along with associated cost.
        self.visitedNodeCombinations = {}
        # List to hold reversible runIds; the names of Abaqus runs.
        self.runTable = []
        # Get a list of all node objects and create a KDTree.
        nodes = []
        for instance in self.model.rootAssembly.instances.values():
            nodes.extend(instance.nodes)
        self.kdTree = AbaqusNodeKDTree(nodes)
        
    def nodesFromVector(self, vector):
        """
        Any given population vector is a string of xyz coordinates, e.g.
        [x1,y1,z1,x2,y2,z2,x3,y3,z3].
        This function splits the vector into a list of tuples, e.g.
        [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)],
        then returns a tuple (hashable) of nearest nodes.
        """
        x, y, z = vector[0::3], vector[1::3], vector[2::3] 
        coordinates = zip(x, y, z)
        nodes = [self.kdTree.query(coord) for coord in coordinates]
        return tuple(nodes)
        
    def getOrCreateRunIdentifier(self, nodes):
        """
        Make a string identifier of the chosen nodes that can be
        reversed at the end of the analysis.
        """
        linearNodesList = ['%s_%s'%(i, l) for i, l in nodes]
        linearNodesString = '_'.join(linearNodesList)
        runIdentifier = 'RUN_%s'%(linearNodesString)
        if runIdentifier not in self.runTable:
            self.runTable.append(runIdentifier)
        index = self.runTable.index(runIdentifier)
        return 'RUN_%s'%(index)
        
    def _cost(self, vector):
        """
        Evaluating the cost function follows these steps:
        - Retrieve the closest nodes for the vector
        - Return the known cost if these nodes have been evaluated before
        - Else, generate a runId, then...
        - return the costFile.cost function, passing the nodes, model and runId as arguments.
        """
        nodes = self.nodesFromVector(member.vector)
        if nodes in self.visitedNodeCombinations:
            return self.visitedNodeCombinations[nodes]
        else:
            runId = self.getOrCreateRunIdentifier(nodes)
            return self.cost(nodes, self.model, runId)
        
    def assignCosts(self, population):
        """
        Change self.cost to the wrapper self._cost.
        """
        costs = map(self._cost, population.vectors)
        self.functionEvaluations += population.size
        for i in xrange(population.size):
            population.members[i].cost = costs[i]
        return population
        
    def terminationCriterion(self):
        """
        Returns True if all of the population have converged on 
        identical nodes - the only sensible FE termination criterion.
        """
        solutions = set()
        for member in self.population.members:
            nodes = self.nodesFromVector(member.vector)
            solutions.add(nodes)
        return len(solutions) == 1
        
    def optimise(self):
        """
        Prepend file I/O operations before running.
        """
        folder = 'DE_' + str(time.time())[:-3]
        os.mkdir(folder)
        os.chdir(folder)
        return super(AbaqusJADE, self).optimise()
