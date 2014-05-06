# For TUOS Iceberg, to make packages like SciPy which are available
# in Python 2.6 available to Abaqus's Python implementation.
import sys
sys.path.append('/usr/lib64/python2.6/site-packages')
# Import Abaqus tools
from abaqus import *
from abaqusConstants import *
from odbAccess import *
# DE code
from jade import JADEWithArchive
# General dependencies
from scipy import spatial
import numpy
import os
import time
import subprocess
from glob import glob


def customPrint(text):
    """
    Print to sys.__stderr__ (like Abaqus does).
    """
    sys.__stderr__.write('%s%s'%(text, os.linesep))
    sys.__stderr__.flush()


class AbaqusNodeKDTree(spatial.KDTree):
    """
    Used to identify the closest node to point in the continuous domain.
    Operates on a list of Abaqus MeshNode objects.
    'query' method returns the instance name and label of a node
    closest to a co-ordinate point (x, y, z).
    """
    def __init__(self, nodeObjects, *args, **kwargs):
        # Split the objects out into two parallel lists:
        self.labels = [(node.instanceName, node.label) for node in nodeObjects]
        data = [node.coordinates for node in nodeObjects]
        super(AbaqusNodeKDTree, self).__init__(data, *args, **kwargs)

    def query(self, x, *args, **kwargs):
        distance, dataIndex = super(AbaqusNodeKDTree, self).query(x, *args, **kwargs)
        return self.labels[dataIndex]


class AbaqusJADE(JADEWithArchive):
    """
    Extensions/changes to JADE for use in Abaqus.
    Everything DE/file IO specific is in this class.
    Everything Abaqus-specific is, as far as is practical,
    stored in the AbaqusProblem class.
    """
    def __init__(self, AbaqusProblemClass, np=12, maxFunctionEvals=50000):
        """
        We need more from the cost file, i.e. a model and instructions to
        manipulate it. Instead, we use a cost class (an AbaqusProblem instance).
        """
        # File I/O operations
        folder = 'DE_' + str(time.time())[:-3]
        os.mkdir(folder)
        os.chdir(folder)
        # Instantiate the problem class
        self.problemClass = AbaqusProblemClass()
        # Assemble kwargs for superclass and call.
        kwargs = {
            'costFile': self.problemClass,
            'np': np,
            'maxFunctionEvals': maxFunctionEvals
        }
        super(AbaqusJADE, self).__init__(**kwargs)
        # A table for visited nodes and their associated costs / unique run ids.
        # Entries are tuples: visitedNodes[x] = (cost, urid)
        self.visitedNodes = {}

    def commutativeVector(self, vector):
        """
        Adjust a vector such that it is sorted by x, y then z.
        """
        x, y, z = vector[0::3], vector[1::3], vector[2::3]
        coordinates = zip(x, y, z)
        coordinates.sort()
        # Flatten back into a Numpy array
        vector = numpy.array([el for coord in coordinates for el in coord])
        return vector

    def runAndWait(self, jobs):
        """
        Run all jobs and wait for completion.
        This function is unique to TUOS Iceberg.
        """
        for urid in jobs:
            os.system('qsub -j y -l h_rt=01:00:00 -l mem=16G -l rmem=8G /usr/local/bin/abaqus611job job=%s interactive cpus=1'%(urid))
        # Wait for completion
        output = 'abaqus'
        while 'abaqus' in output:
            time.sleep(1)
            p = subprocess.Popen('Qstat', stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            output, errors = p.communicate()

    def cleanFolder(self):
        """
        Remove non-ODB files from working directory
        """
        for f in glob('*'):
            if 'odb' not in f:
                os.remove(f)

    def assignCosts(self, population):
        """
        Compute and assign cost function values to each member of the passed
        population.Population instance by considering the member vectors.
        Return the modified population.
        """
        # Sort population vectors.
        for i, member in enumerate(population.members):
            population.members[i].vector = self.commutativeVector(member.vector)
        # Prepare a to-do list of runs
        toDo = {}
        for i, member in enumerate(population.members):
            nodes = self.problemClass.getNodesFromVector(member.vector)
            population.members[i].nodes = nodes
            # Check for uniqueness of nodes. Allocate penalty cost if neccesary.
            if len(set(nodes)) != self.problemClass.numberOfNodes:
                self.visitedNodes[nodes] = (numpy.inf, None)
            elif (nodes not in toDo) and (nodes not in self.visitedNodes):
                urid = 'de_%s'%(self.functionEvaluations)
                self.functionEvaluations += 1
                toDo[nodes] = urid
        # Process, write input for and run the to-do list (async)
        for nodes, urid in toDo.iteritems():
            model = self.problemClass.getModelCopy(urid)
            testModel = self.problemClass.setUp(nodes, urid, model)
            self.problemClass.writeInput(testModel, urid)
            self.problemClass.tearDown()
        self.runAndWait(toDo.values())
        # Evaluate once complete.
        for nodes, urid in sorted(toDo.iteritems()):
            odb = self.problemClass.getOdb('%s.odb'%(urid))
            # There is a small chance that the analysis may have failed due to a
            # licence or memory error. Skip this case.
            try:
                cost = self.problemClass.postProcess(odb)
                odb.close()
            except:
                cost = numpy.inf
                customPrint('Aborting %s'%(urid))
            self.visitedNodes[nodes] = (cost, urid)
        # Assign costs and unique run ids to members based on nodes
        for i, member in enumerate(population.members):
            population.members[i].cost = self.visitedNodes[member.nodes][0]
            population.members[i].urid = self.visitedNodes[member.nodes][1]
        return population

    def terminationCriterion(self):
        """
        Returns True if all of the population have converged on
        identical nodes - the only sensible FE termination criterion.
        Also print logging info to stderr, and conduct file I/O operations.
        """
        # Print logging info
        best = self.population.members[self.population.bestVectorIndex]
        worst = self.population.members[self.population.worstVectorIndex]
        customPrint('At generation %s'%(self.generation))
        customPrint('  Best-so-far vector: %s'%(best))
        customPrint('  Worst population vector: %s'%(worst))
        customPrint('  URID of best vector: %s'%(best.urid))
        customPrint('  URIDs in population: ' + \
                ', '.join(sorted(list(set(x.urid for x in self.population.members if x.urid is not None)))))
        customPrint('  Standard Deviation of vectors: %s'%(self.population.standardDeviation))
        customPrint('  Mean control parameters: f=%s, cr=%s'%(self.f, self.cr))
        # File I/O
        self.cleanFolder()
        # Termination criteria
        solutions = set([member.nodes for member in self.population.members])
        return len(solutions) == 1


class AbaqusProblem(object):
    """
    An abstract class to be subclassed in specific Abaqus optimisation problems.

    AbaqusProblems must implement methods to get a model, set up an analysis and
    post-process.

    Methods to run write input files are provided, as is a method of translating a
    continuous vector to a sequence of nearest nodes (using a KD-Tree).
    A tearDown() method is called at the end of the run.
    """
    numberOfNodes = 1
    absoluteBounds = False

    def __init__(self):
        self.baseModel = self.getBaseModel()
        # Get a list of node objects and create a KDTree.
        self.nodes = self.getFeasibleNodes(self.baseModel)
        self.kdTree = AbaqusNodeKDTree(self.nodes)

    def getBaseModel(self):
        """
        Subclasses must include this method. It should return an mdb.Model instance.
        """
        raise NotImplementedError

    def getFeasibleNodes(self, model):
        """
        Subclasses must include this method. It should return a list of MeshNode
        objects representing the feasible region.
        """
        raise NotImplementedError

    def getModelCopy(self, urid):
        """
        Return a copy of self.baseModel
        """
        return mdb.Model(name=urid, objectToCopy=self.baseModel)

    def getBounds(self):
        """
        Infer the bounds from the problem dimensions.
        """
        coords = [node.coordinates for node in self.nodes]
        minimum = numpy.min(coords, axis=0)
        maximum = numpy.max(coords, axis=0)
        n = self.numberOfNodes
        return n * list(minimum), n * list(maximum)

    def setUp(self, nodes, model, urid):
        """
        Based on a sequence of nodes passed, return a model that it is ready to be run.
        """
        raise NotImplementedError

    def tearDown(self):
        """
        Called after each run. Include any actions that need to be performed on
        the base model to return it to its as-initialised state.
        """
        for name in mdb.models.keys():
            if name != self.baseModel.name:
                del mdb.models[name]

    def getOdb(self, odbName):
        """
        Try to get an ODB from a file name.
        """
        try:
            return openOdb(path=odbName)
        except:
            return None

    def postProcess(self, odb):
        """
        Should return a scalar cost for the passed ODB.
        """
        raise NotImplementedError

    def writeInput(self, model, urid):
        """
        Create an analysis job for the model and write an input file.
        """
        job = mdb.Job(name=urid, model=model)
        job.writeInput()
        del mdb.jobs[urid]

    def getNodesFromVector(self, vector):
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

    def cost(self, x):
        """
        Here only to provide API consistency with continuous DE.
        Will be ignored by Abaqus DE.
        """
        pass
