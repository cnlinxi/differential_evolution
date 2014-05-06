# Abaqus imports
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import regionToolset
import interaction
# DE
from deAbaqus import AbaqusProblem, AbaqusJADE
# Generic
import os


class BloodhoundProblem(AbaqusProblem):

    numberOfNodes = 16
    
    def getBaseModel(self):
        f = openMdb('abaqusFiles/bloodhound.cae')
        model = f.models['Bloodhound_Baffle']
        return model
        
    def getFeasibleNodes(self, model):
        return model.rootAssembly.sets['edges'].nodes
        
    def setUp(self, nodes, urid, model):
        """
        Add bolted joints at 'nodes'
        """
        TANK_THICKNESS = 13.2
        extantSteps = model.steps.keys()
        # New step after the last step
        model.StaticStep(urid, extantSteps[-1])
        a = model.rootAssembly
        for (nodeInstance, nodeLabel) in nodes:
            nodeName = '%s_%s'%(nodeInstance, nodeLabel)
            instance = model.rootAssembly.instances[nodeInstance]
            nodeSequence = instance.nodes.sequenceFromLabels([nodeLabel])
            node = nodeSequence[0]
            coords = node.coordinates
            # Create a reference point at the node
            nodeRef = a.ReferencePoint(point=coords)
            # Establish where the node is on the baffle (top/bottom/left/right/dia).
            for name, s in a.sets.items():
                if node in s.nodes and name != 'edges':
                    nodeSetName = name
                    break
            # Create a second reference point accordingly
            if 'above' in nodeSetName:
                anchorPoint = (coords[0], coords[1] + TANK_THICKNESS, coords[2])
                csysId = a.features['csys-vertical'].id
            elif 'below' in nodeSetName:
                anchorPoint = (coords[0], coords[1] - TANK_THICKNESS, coords[2])
                csysId = a.features['csys-vertical'].id
            elif 'dia-left' in nodeSetName:
                anchorPoint = (coords[0] - (TANK_THICKNESS / (2**0.5)), 
                        coords[1] - (TANK_THICKNESS / (2**0.5)), coords[2])
                csysId = a.features['csys-dia-left'].id
            elif 'dia-right' in nodeSetName:
                anchorPoint = (coords[0] + (TANK_THICKNESS / (2**0.5)), 
                        coords[1] - (TANK_THICKNESS / (2**0.5)), coords[2])
                csysId = a.features['csys-dia-right'].id
            elif 'left' in nodeSetName:
                anchorPoint = (coords[0] -  TANK_THICKNESS, coords[1], coords[2])
                csysId = a.features['csys-horizontal'].id
            elif 'right' in nodeSetName:
                anchorPoint = (coords[0] +  TANK_THICKNESS, coords[1], coords[2])
                csysId = a.features['csys-horizontal'].id            
            anchorRef = a.ReferencePoint(point=anchorPoint)
            # Create a wire feature for the connector
            wire = a.WirePolyLine(points=((a.referencePoints[nodeRef.id], 
                    a.referencePoints[anchorRef.id]), ), mergeWire=OFF, meshable=OFF)
            # Find the length-1 EdgeArray corresponding to this wire
            i = 0
            while a.edges[i].featureName != wire.name:
                i += 1
            wireSet = a.Set(edges=[a.edges[i:i+1], ], name=nodeName)
            # Assign to M4 bolt with correct orientation
            csa = a.SectionAssignment(sectionName='M4', region=wireSet)
            a.ConnectorOrientation(region=csa.getSet(), localCsys1=a.datums[csysId])
            # Encastre the anchor point
            reg = regionToolset.Region(referencePoints=(a.referencePoints[anchorRef.id], ))
            model.EncastreBC(name=nodeName, createStepName=urid, region=reg)
            # Couple the nodal reference point to the node itself
            referenceRegion = regionToolset.Region(referencePoints=(a.referencePoints[nodeRef.id], ))
            nodeRegion = regionToolset.Region(nodes=nodeSequence)
            model.Coupling(name=nodeName, controlPoint=referenceRegion, surface=nodeRegion, 
                    influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None,
                    u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        # Deactivate old BC and return
        model.boundaryConditions['fixed_sides'].deactivate(urid)
        return model
    
    def postProcess(self, odb):
        """
        Get the maximum deflection in the baffle
        """
        finalStep = odb.steps.values()[-1]
        finalFrame = finalStep.frames[-1]
        deflectionField = finalFrame.fieldOutputs['U']
        peak = max([u.magnitude for u in deflectionField.values])
        return peak
        
        
if __name__ == '__main__': 
    problem = AbaqusJADE(AbaqusProblemClass=BloodhoundProblem, np=40)
    bestVector = problem.optimise()
    sys.__stderr__.write('BEST: %s%s'%(bestVector, os.linesep))
    sys.__stderr__.flush()
