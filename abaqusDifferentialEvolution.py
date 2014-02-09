from abaqus import *
from abaqusConstants import *
from odbAccess import *
from scipy import spatial
import numpy
import os
import time
from fyp import differentialEvolution

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
              
        
class AbaqusDifferentialEvolution(differentialEvolution.DifferentialEvolution):
    '''
    Modifications/Extensions to the DifferentialEvolution class 
    to make it appropriate for use in an Abaqus CAE script.
    '''
    def __init__(self):
        folder = 'DE_' + str(time.time())[:-3]
        os.mkdir(folder)
        os.chdir(folder)
        self.model = self.getModel()
        self.kdTree = self.makeKDTree()
        self.visitedNodeCombinations = {}
        self.runTable = []
        super(AbaqusDifferentialEvolution, self).__init__()
        # Change the default convergence function.
        self.convergence_function = self.commonNodeConvergence
        # c should be high as dimensions are generally not separable.
        self.c = 1
        
    def splitIntoCoordinates(self, vector):
        '''
        Any given population vector is a string of xyz coordinates, e.g.
        [x1,y1,z1,x2,y2,z2,x3,y3,z3].
        This function splits the vector into a list of tuples, e.g.
        [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
        '''
        x = vector[0::3]
        y = vector[1::3]
        z = vector[2::3]
        return zip(x, y, z)
        
    def nodesFromVector(self, vector):
        '''
        Given a population vector as described in splitIntoCoordinates,
        return a tuple (hashable) of nearest nodes.
        '''
        coordinates = self.splitIntoCoordinates(vector)
        nodes = [self.kdTree.query(coord) for coord in coordinates]
        return tuple(nodes)
        
    def commonNodeConvergence(self):
        '''
        Returns True if all of the population have converged on 
        identical nodes.
        '''
        solutions = set()
        for vector in self.population:
            nodes = self.nodesFromVector(vector)
            solutions.add(nodes)
        return len(solutions) == 1
        
    def getOrCreateRunIdentifier(self, nodes):
        '''
        Make a string identifier of the chosen nodes that can be
        reversed at the end of the analysis.
        '''
        linearNodesList = ['%s_%s'%(i, l) for i, l in nodes]
        linearNodesString = '_'.join(linearNodesList)
        runIdentifier = 'RUN_%s'%(linearNodesString)
        if runIdentifier not in self.runTable:
            self.runTable.append(runIdentifier)
        index = self.runTable.index(runIdentifier)
        return 'RUN_%s'%(index)
        
    def getModel(self):
        '''
        Should return an AbaqusModel object which is
        ready to be run.
        '''
        raise NotImplementedError
        
    def makeKDTree(self):
        '''
        Get a list of all node objects and create a KDTree.
        '''
        nodes = []
        for instance in self.model.rootAssembly.instances.values():
            nodes.extend(instance.nodes)
        return AbaqusNodeKDTree(nodes)
        
    def runJob(self, jobName):
        '''
        Create an analysis job for the model and submit it.
        Will probably be used in cost functions.
        '''
        myJob = mdb.Job(name=jobName, model=self.model.name,
            description=self.model.name)
        # Wait for the job to complete.
        # Conditional logic to help survive a license error.
        jobSuccess = False
        differentialEvolution.printer('\nRun commencing...')
        while not jobSuccess:
            try:
                myJob.submit()
                myJob.waitForCompletion()
                # Open the output database
                self.odb = openOdb(path=jobName + '.odb')
                jobSuccess = True
            except:
                del mdb.jobs[jobName]
                myJob = mdb.Job(name=jobName, model=self.model.name,
                    description=self.model.name)
        
    def solution_complete_hook(self, victor):
        '''
        Open the 'winning' ODB
        '''
        nodes = self.nodesFromVector(victor)
        runIdentifier = self.getOrCreateRunIdentifier(nodes)
        os.system('abaqus viewer odb=%s'%(runIdentifier))
        return super(AbaqusDifferentialEvolution, self).solution_complete_hook(victor)
     