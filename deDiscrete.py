from abaqus import *
from abaqusConstants import *
from odbAccess import *
import regionToolset
import sys; sys.path.append('/usr/lib64/python2.6/site-packages')
from scipy import spatial
import multiprocessing
import numpy
import os
import time
import copy
from deAbaqus import AbaqusJADE
            
            
class AbaqusNodeKDTree(spatial.KDTree):
    """
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
        self.baseMdb = self.getMDB()
        # Get a list of all node objects and create a KDTree.
        nodes = []
        for instance in self.baseMdb.rootAssembly.instances.values():
            nodes.extend(instance.nodes)
        self.kdTree = AbaqusNodeKDTree(nodes)
        
    def getMDB(self):
        """
        Subclasses must include this method. It should return an Abaqus MDB.
        """
        raise NotImplementedError
        
    def setUp(self, nodes, urid):
        """
        Based on a sequence of nodes passed, return an mdb based on self.mdb
        that it is ready to be run.
        """
        raise NotImplementedError
        
    def tearDown(self):
        """
        Called after each run. Include any actions that need to be performed on
        self.mdb to return it to its as-initialised state.
        """
        pass
        
    def postProcess(self, odbName):
        """
        Should return a scalar cost for the passed ODB filename.
        """
        raise NotImplementedError
        
    def writeInput(self, testMdb, urid):
        """
        Create an analysis job for the model and submit it.
        Return an odb object.
        """
        job = mdb.Job(name=urid, model=testMdb)
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
                    
    def cost(self, nodes):
        """
        A generic cost function, suitable for most analyses.
        """
        pass
        
    
# Tests

def buildABeam(name, length, materialName, youngsModulus, 
    poissonsRatio, width, height, numberOfElements, loadMagnitude):
    '''
    A function to build an mdb.Model object for a beam,
    with loading.
    '''
    model = mdb.Model(name=name)
    # Create a sketch for the base feature.
    sketch = model.ConstrainedSketch(name=model.name, sheetSize=250.)
    # Create a line (the beam centreline).
    length = float(length)
    sketch.Line(point1=(0.0, 0.0), point2=(length, 0.0))
    # Create a two-dimensional, deformable part.
    beamPart = model.Part(name='Beam', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    beamPart.BaseWire(sketch=sketch)
    # Create a material.
    material = model.Material(name=materialName)
    # Create the elastic properties.
    elasticProperties = (youngsModulus, poissonsRatio)
    material.Elastic(table=(elasticProperties, ) )
    # Create a rectangular beam section.
    profile = model.RectangularProfile(name='beamProfile', a=height, b=width)
    section = model.BeamSection(name='beamSection', profile='beamProfile', 
        integration=DURING_ANALYSIS, poissonRatio=0.0, material=material.name, 
        temperatureVar=LINEAR)
    # Assign the section to the region. The region refers 
    # to the single cell in this model.
    region = (beamPart.edges,)
    beamPart.SectionAssignment(region=region, sectionName='beamSection')
    beamPart.assignBeamSectionOrientation(region=region, 
        method=N1_COSINES, n1=(0.0, 0.0, 1.0))
    # Create a part instance.
    assembly = model.rootAssembly
    instance = assembly.Instance(name='beamInstance',
        part=beamPart, dependent=OFF)
    # Assign an element type to the part instance.
    region = (instance.edges,)
    #elemType = mesh.ElemType(elemCode=B21, elemLibrary=STANDARD)
    assembly = model.rootAssembly
    #assembly.setElementType(regions=region, elemTypes=(elemType,))
    # Seed the part instance.
    seedSize = length / numberOfElements
    assembly.seedPartInstance(regions=(instance,), size=seedSize)
    # Mesh the part instance.
    assembly.generateMesh(regions=(instance,))
    # Create a step to define the loading and boundary conditions.
    model.StaticStep(name='beamLoad', previous='Initial')
    # Find the end point by coordinates.
    endVertexLocation = (0.0, 0.0, 0.0)
    endVertex = instance.vertices.findAt((endVertexLocation,) )
    # Create a boundary condition that encastres one end
    # of the beam.
    endRegion = (endVertex,)
    model.EncastreBC(name='beamLoad',createStepName='beamLoad',
        region=endRegion)
    # Find the top edge using coordinates.
    topEdge = instance.edges.findAt((endVertexLocation,) )
    # Create a pressure load on the top edge of the beam.
    topSurface = ((topEdge, SIDE1), )
    model.Pressure(name='Pressure', createStepName='beamLoad',
        region=topSurface, magnitude=loadMagnitude)
    return model
    
    
length = 500

class BeamProblem(AbaqusProblem):
    
    def getMDB(self):
        return buildABeam('beam', length, 'steel', 210000, 0.3, 10, 10, 100, 1)
        
    def getBounds(self):
        n = 1  # Number of encastres
        return [0, 0, 0] * n, [length, 0, 0] * n
        
    def setUp(self, nodes, urid):
        """
        Create a new mdb with "nodes" constrained on the last step.
        """
        testMdb = self.baseMdb
        extantSteps = testMdb.steps.keys()
        testMdb.StaticStep(urid, extantSteps[-1])
        for (nodeInstance, nodeLabel) in nodes:
            bcName = 'ENCASTRE_%s_%s'%(nodeInstance, nodeLabel)
            instance = testMdb.rootAssembly.instances[nodeInstance]
            nodeSequence = instance.nodes.sequenceFromLabels([nodeLabel])
            region = regionToolset.Region(nodes=nodeSequence)
            testMdb.EncastreBC(name=bcName, createStepName=urid, region=region)
            testMdb.boundaryConditions[extantSteps[-1]].deactivate(urid)
        return testMdb

    def tearDown(self):
        keys = self.baseMdb.steps.keys()
        del self.baseMdb.steps[keys[-1]]
    
    def postProcess(self, odbName):
        """
        Get the maximum deflection in the beam
        """
        odb = openOdb(path=odbName)
        finalStep = odb.steps.values()[-1]
        finalFrame = finalStep.frames[-1]
        deflectionField = finalFrame.fieldOutputs['U']
        peak = max([u.magnitude for u in deflectionField.values])
        odb.close()
        return peak
             
if __name__ == '__main__': 
    problem = AbaqusJADE(ProblemClass=BeamProblem)
    bestVector = problem.optimise()
    sys.__stderr__.write('BEST: %s%s'%(bestVector, os.linesep))
    sys.__stderr__.flush()
