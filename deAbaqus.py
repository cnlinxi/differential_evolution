from jade import JADEWithArchive
import numpy
import os
import sys
import time
import subprocess

class AbaqusJADE(JADEWithArchive):
    """
    Extensions/changes to JADE for use in Abaqus.
    """
    def __init__(self, ProblemClass, np=12, maxFunctionEvals=50000):
        """
        We need more from the cost file, i.e. a model and constraints.
        From this model, create a KDTree instance.
        """
        # File I/O operations
        folder = 'DE_' + str(time.time())[:-3]
        os.mkdir(folder)
        os.chdir(folder)
        # Instantiate the problem class
        self.problemClass = ProblemClass()
        # Assemble kwargs for superclass and call.
        kwargs = {
            'costFile': self.problemClass,
            'np': np, 
            'maxFunctionEvals': maxFunctionEvals
        }
        super(AbaqusJADE, self).__init__(**kwargs)
        # Additional requirements: equality and inequality constraints
        # self.constraints = problemClass.getConstraints()
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
        # Process, write input and run the to-do list (async)
        for nodes, urid in toDo.iteritems():
            testMdb = self.problemClass.setUp(nodes, urid)
            self.problemClass.writeInput(testMdb, urid)
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
            odb = '%s.odb'%(urid)
            time.sleep(5)
            cost = self.problemClass.postProcess(odb)
            sys.__stderr__.write(str(cost) + os.linesep)
            sys.__stderr__.flush()
            self.visitedNodes[nodes] = (cost, urid)
        # Assign costs to members based on nodes
        for i, member in enumerate(population.members):
            population.members[i].cost = self.visitedNodes[member.nodes][0]
        return population

    def terminationCriterion(self):
        """
        Returns True if all of the population have converged on 
        identical nodes - the only sensible FE termination criterion.
        """
        solutions = set([member.nodes for member in self.population.members])
        return len(solutions) == 1
