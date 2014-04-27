# Abaqus imports
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import regionToolset
# DE
from deAbaqus import AbaqusProblem, AbaqusJADE
# Generic
import os
import time
        
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
    

class BeamProblem(AbaqusProblem):
    
    def getBaseModel(self):
        return buildABeam('beam', 1000, 'steel', 210000, 0.3, 10, 10, 500, 1)
        
    def getFeasibleNodes(self, model):
        nodes = []
        for instance in model.rootAssembly.instances.values():
            nodes.extend(instance.nodes)
        return nodes
        
    def getBounds(self):
        minimum, maximum = super(BeamProblem, self).getBounds()
        n = 3  # Number of encastres
        return minimum * n, maximum * n
        
    def setUp(self, nodes, urid, model):
        """
        Create a new mdb with "nodes" constrained on the last step.
        """
        extantSteps = model.steps.keys()
        model.StaticStep(urid, extantSteps[-1])
        for (nodeInstance, nodeLabel) in nodes:
            bcName = 'ENCASTRE_%s_%s'%(nodeInstance, nodeLabel)
            instance = model.rootAssembly.instances[nodeInstance]
            nodeSequence = instance.nodes.sequenceFromLabels([nodeLabel])
            region = regionToolset.Region(nodes=nodeSequence)
            model.EncastreBC(name=bcName, createStepName=urid, region=region)
            model.boundaryConditions[extantSteps[-1]].deactivate(urid)
        return model
    
    def postProcess(self, odb):
        """
        Get the maximum deflection in the beam
        """
        finalStep = odb.steps.values()[-1]
        finalFrame = finalStep.frames[-1]
        deflectionField = finalFrame.fieldOutputs['U']
        peak = max([u.magnitude for u in deflectionField.values])
        return peak
             
if __name__ == '__main__': 
    problem = AbaqusJADE(AbaqusProblemClass=BeamProblem, np=15)
    bestVector = problem.optimise()
    sys.__stderr__.write('BEST: %s%s'%(bestVector, os.linesep))
    sys.__stderr__.flush()
