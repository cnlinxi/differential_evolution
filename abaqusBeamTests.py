from abaqus import *
from abaqusConstants import *
# import mesh
import regionToolset
import time
from fyp import differentialEvolution, abaqusDifferentialEvolution


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
    
    
def getDeflections(odb):
    deflections = {}
    steps = odb.steps.values()
    finalFrames = [step.frames[-1] for step in steps]
    minTipDeflection = None
    for frame in finalFrames:
        outputs = frame.fieldOutputs['U']
        magnitudes = []
        for output in outputs.values:
            magnitude = output.magnitude
            if float(magnitude) < 10**-10:
                constrainedNode = str(output.nodeLabel)
            else:
                magnitudes.append(magnitude)
        tipDeflection = max(magnitudes)
        if not minTipDeflection or tipDeflection < minTipDeflection:        
            minTipDeflection = tipDeflection
            node = constrainedNode
    return node, minTipDeflection

    
class AbaqusBeamDifferentialEvolution(abaqusDifferentialEvolution.AbaqusDifferentialEvolution):

    def __init__(self):
        self.length = 5000
        super(AbaqusBeamDifferentialEvolution, self).__init__()
        self.absolute_bounds = True
        self.population_size = 8
        self.inequality_constraints = [None,None,None,('gte',0),None,None,('gte',3),None,None]
        
    def getModel(self):
        return buildABeam('beam', self.length, 'steel', 210000, 0.3, 10, 10, 100, 1)
    
    def cost(self, x):
        m = self.model
        extantSteps = m.steps.keys()
        nodes = self.nodesFromVector(x)
        if nodes not in self.visitedNodeCombinations:
            runIdentifier = self.getOrCreateRunIdentifier(nodes) # Also used for step name
            m.StaticStep(runIdentifier, extantSteps[-1])
            for (nodeInstance, nodeLabel) in nodes:
                bcName = 'ENCASTRE_%s_%s'%(nodeInstance, nodeLabel)
                instance = m.rootAssembly.instances[nodeInstance]
                nodeSequence = instance.nodes.sequenceFromLabels([nodeLabel])
                region = regionToolset.Region(nodes=nodeSequence)
                m.EncastreBC(name=bcName, createStepName=runIdentifier, region=region)
            m.boundaryConditions[extantSteps[-1]].deactivate(runIdentifier)
            self.runJob(runIdentifier)
            node, tipDeflection = getDeflections(self.odb)
            self.visitedNodeCombinations[nodes] = tipDeflection
            # Tidy up
            self.odb.close()
            del m.steps[runIdentifier]
            del mdb.jobs[runIdentifier]
        else:
            differentialEvolution.printer('\nData for node combination %s is being read from the cache.\n'%(str(nodes)))
            tipDeflection = self.visitedNodeCombinations[nodes]
            self.function_evaluations -= 1
        return tipDeflection
        
    def get_bounding_vectors(self):
        n = 3  # Number of encastres
        return [0, 0, 0] * n, [self.length, 0, 0] * n
        
        
start = time.clock()        
problem = AbaqusBeamDifferentialEvolution()
differentialEvolution.printer(problem.solve())
differentialEvolution.printer('Execution time: %s seconds\n'%(time.clock() - start))