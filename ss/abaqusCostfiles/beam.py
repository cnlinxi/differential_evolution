from abaqus import *
from abaqusConstants import *
import regionToolset


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

    
length = 5000
absoluteBounds = True
        
def getModel():
    return buildABeam('beam', length, 'steel', 210000, 0.3, 10, 10, 100, 1)
    
def getConstraints():
    return [None,None,None,('gte',0),None,None,('gte',3),None,None]
    
def runJob(jobName):
    """
    Create an analysis job for the model and submit it.
    Will probably be used in cost functions.
    """
    myJob = mdb.Job(name=jobName, model=self.model.name,
        description=self.model.name)
    # Wait for the job to complete.
    # Conditional logic to help survive a license error.
    jobSuccess = False
    print '\nRun commencing...'
    while not jobSuccess:
        try:
            myJob.submit()
            myJob.waitForCompletion()
            # Open the output database
            return openOdb(path=jobName + '.odb')
            jobSuccess = True
        except:
            del mdb.jobs[jobName]
            myJob = mdb.Job(name=jobName, model=self.model.name,
                description=self.model.name)
    
def cost(self, nodes, m, runIdentifier):
    """
    For Abaqus optimisations, nodes (as opposed to vectors) are passed.
    Additional arguments are a model, m, and a runIdentifier
    which should be used for the step name and run.
    """
    # M is model
    extantSteps = m.steps.keys()
    m.StaticStep(runIdentifier, extantSteps[-1])
    for (nodeInstance, nodeLabel) in nodes:
        bcName = 'ENCASTRE_%s_%s'%(nodeInstance, nodeLabel)
        instance = m.rootAssembly.instances[nodeInstance]
        nodeSequence = instance.nodes.sequenceFromLabels([nodeLabel])
        region = regionToolset.Region(nodes=nodeSequence)
        m.EncastreBC(name=bcName, createStepName=runIdentifier, region=region)
    m.boundaryConditions[extantSteps[-1]].deactivate(runIdentifier)
    odb = runJob(runIdentifier)
    node, tipDeflection = getDeflections(self.odb)
    # Tidy up
    self.odb.close()
    del m.steps[runIdentifier]
    del mdb.jobs[runIdentifier]
    return tipDeflection
    
def getBounds(self):
    n = 3  # Number of encastres
    return [0, 0, 0] * n, [length, 0, 0] * n