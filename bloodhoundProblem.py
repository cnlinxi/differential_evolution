# Abaqus imports
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import regionToolset
# DE
from deAbaqus import AbaqusProblem, AbaqusJADE
# Generic
import os


class BloodhoundProblem(AbaqusProblem):

    numberOfNodes = 12
    
    def getBaseModel(self):
        f = openMdb('../bloodhound.cae')
        model = f.models['Bloodhound_Baffle']
        return model
        
    def getFeasibleNodes(self, model):
        return model.rootAssembly.sets['edges'].nodes
        
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
            model.boundaryConditions['fixed_sides'].deactivate(urid)
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
    problem = AbaqusJADE(AbaqusProblemClass=BloodhoundProblem, np=24)
    bestVector = problem.optimise()
    sys.__stderr__.write('BEST: %s%s'%(bestVector, os.linesep))
    sys.__stderr__.flush()
