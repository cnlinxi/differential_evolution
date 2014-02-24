import testFunctions
from de import DifferentialEvolution2 as DE
import numpy


s = DE(testFunctions.sphere, dimensions=2, f=0.9, fRandomiser='dither',
    multiprocessing=False, huddling=True, verbosity=0)

unhuddled, huddled = [], []
n = 2
runs= 3
for i in xrange(runs):
    s = DE(testFunctions.sphere, dimensions=n, mutator='de/best/1/bin',
        multiprocessing=False, huddling=False, verbosity=0)
    population, evals, gen = s.optimise()
    unhuddled.append(evals)
    print population.bestVector, evals
for i in xrange(runs):
    s = DE(testFunctions.sphere, dimensions=n,mutator='de/best/1/bin',
        multiprocessing=False, huddling=True, verbosity=0)
    population, evals, gen = s.optimise()
    huddled.append(evals)
    print population.bestVector, evals

    
print 'Huddle factor: %s'%(numpy.mean(huddled) / numpy.mean(unhuddled))