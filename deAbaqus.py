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
        
    def assignCosts(self, population):
        """
        Compute and assign cost function values to each member of the passed 
        population.Population instance by considering the member vectors. 
        Return the modified population.
        """
        # Prepare a to-do list of runs
        toDo = {}
        for i, member in enumerate(population.members):
            nodes = self.problemClass.getNodesFromVector(member.vector)
            population.members[i].nodes = nodes
            if (nodes not in toDo) and (nodes not in self.visitedNodes):
                urid = 'de_%s'%(self.functionEvaluations)
                self.functionEvaluations += 1
                toDo[nodes] = urid
        # Process, write input for and run the to-do list (async)
        for nodes, urid in toDo.iteritems():
            model = self.problemClass.getModelCopy(urid)
            testModel = self.problemClass.setUp(nodes, urid, model)
            self.problemClass.writeInput(testModel, urid)
            self.problemClass.tearDown()
            # THIS LINE IS UNIQUE TO TUOS ICEBERG!
            os.system('qsub -j y -l h_rt=01:00:00 -l mem=12G /usr/local/bin/abaqus611job job=%s interactive cpus=1'%(urid))
        # Wait for completion
        output = 'abaqus'
        while 'abaqus' in output:
            time.sleep(2)
            # THIS LINE IS ALSO UNIQUE TO TUOS ICEBERG!
            p = subprocess.Popen('Qstat',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            output, errors = p.communicate()
        # Evaluate once complete.
        for nodes, urid in sorted(toDo.iteritems()):
            odb = self.problemClass.getOdb('%s.odb'%(urid))
            cost = self.problemClass.postProcess(odb)
            odb.close()
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
        """
        solutions = set([member.nodes for member in self.population.members])
        return len(solutions) == 1
        
                 
class AbaqusProblem(object):
    """
    An abstract class to be subclassed in specific Abaqus optimisation problems.
    
    In addition to the standard methods provided by a continuous-domain
    costFile (cost(x), getBounds()), subclasses of this class should incorporate
    a getNodesFromVector(x) method to translate points in the continuous
    domain to the discrete domain. Bounds are assumed to be absolute unless
    otherwise specified.
    
    AbaqusProblems must also implement methods to get an MDB, set up an analysis and
    post-process (getMDB(), setUp(nodes), and postProcess(odb) respectively).

    A runAnalysis method is provided, as is a method of translating a
    continuous vector to a sequence of nearest nodes (using a KD-Tree).
    A tearDown() method is called at the end of the run, which does nothing in this 
    base class.
    """
    def __init__(self):
        self.absoluteBounds = True
        self.baseModel = self.getModel()
        # Get a list of all node objects and create a KDTree.
        nodes = []
        for instance in self.baseMdb.rootAssembly.instances.values():
            nodes.extend(instance.nodes)
        self.kdTree = AbaqusNodeKDTree(nodes)
        
    def getBaseModel(self):
        """
        Subclasses must include this method. It should return an mdb.Model instance.
        """
        raise NotImplementedError
        
    def getModelCopy(self, urid):
        """
        Return a copy of self.baseModel
        """
        return mdb.Model(name=urid, objectToCopy=self.baseModel)
        
    def setUp(self, nodes, model, urid):
        """
        Based on a sequence of nodes passed, return a model that it is ready to be run.
        """
        raise NotImplementedError
        
    def tearDown(self):
        """
        Called after each run. Include any actions that need to be performed on
        self.baseModel to return it to its as-initialised state.
        """
        pass
        
    def getOdb(self, odbName):
        """
        Get an ODB from a file name.
        """
        return openOdb(path=odbName)
        
    def postProcess(self, odb):
        """
        Should return a scalar cost for the passed ODB.
        """
        raise NotImplementedError
        
    def writeInput(self, model, urid):
        """
        Create an analysis job for the model and submit it.
        Return an odb object.
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
        
