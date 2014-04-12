import numpy, openpyxl, string, testFunctions
from openpyxl.cell import get_column_letter
from differentialEvolution import LoggingDifferentialEvolution as DE, NotConvergedException
ALPHABET = string.uppercase
import time
import itertools
import os
import csv

'''
This file tests the various options of the DifferentialEvolution class by
running them repeatedly on the test functions defined in testFunctions folder.

The results are exported to Microsoft Excel.
'''

def tests():
    dimensions = [2,]
    variables = {
        'populationSize': [5, 10, 20, 40, 80],
        #'f': [None, 0.3, 0.6, 0.9],
        #'cr': [0.1, 0.9],
        #'learningPeriod': [1, 2, 5],
        #'mutator': ['de/rand/1/bin', 'de/best/1/bin', 'de/current_to_best/1/bin', ],#'de/rand/2/bin', 'de/best/2/bin', 'de/rand_then_best/1/bin'],
        #'baseVectorSelector': ['random', 'permuted', 'offset'],
        #'fRandomiser': ['static', 'dither', 'jitter', 'randomInterval'],
        #'huddling': [True, False]
    }
    repeats = 10
    # Initialise Excel workbook
    wb = openpyxl.Workbook()
    wb_name = 'DE_Tests_%s.xlsx'%(time.strftime('%d-%m-%Y__%H:%M'))
    worksheets = {}
    # Run the tests
    for d in dimensions:
        unimodal_problems = {
            #'sphere': testFunctions.sphere,
            'hyper-ellipsoid': testFunctions.hyperellipsoid,
            'rozenbrock': testFunctions.rozenbrock,
            'schwefel-ridge': testFunctions.schwefelridge,
            #'neumaier': testFunctions.neumaier, # NONZERO TARGET
        }
        multimodal_problems = {
            'ackley': testFunctions.ackley,
            'griewangk': testFunctions.griewangk,
            'rastrigin': testFunctions.rastrigin,
            'salomon': testFunctions.salomon,
            'whitley': testFunctions.whitley,
            #'storn': testFunctions.Storn,
            #'lennard-jones': testFunctions.LennardJones,
            #'hilbert': testFunctions.Hilbert,
            #'modified-langerman': testFunctions.ModifiedLangerman,
            #'shekel': testFunctions.Shekel,
            #'odd-square': testFunctions.OddSquare,
            #'katsuura': testFunctions.Katsuura,
        }
        bound_problems = {
            #'schwefel': testFunctions.schwefel, # NONZERO TARGET
            #'michalewicz': testFunctions.Michalewicz,
            #'rana': testFunctions.rana,  # NONZERO TARGET
            # 'beam': testFunctions.beam  # NONZERO TARGET
        }
        all_problems = dict(unimodal_problems.items() +
            multimodal_problems.items() + bound_problems.items())
        for problem_name, problem in sorted(all_problems.iteritems()):
            problem_descr = '%s in %s dimensions'%(problem_name, d)
            print 'Testing %s'%(problem_descr)
            for var_name, var_list in sorted(variables.iteritems()):
                for attr in var_list:
                    short_var_name = var_name.replace('/', '')
                    if len(short_var_name) > 5:
                        short_var_name = var_name[:5]
                    attr_name = str(attr).replace('/', '')
                    if len(attr_name) > 10:
                        attr_name = attr_name[-10:]
                    problem_id = '%s_%sD_%s=%s'%(problem_name[:5], d, short_var_name, attr_name)
                    print problem_id
                    worksheets[problem_id] = wb.create_sheet()
                    ws = worksheets[problem_id]
                    ws.title = problem_id
                    failures = 0
                    kwargs = {var_name: attr}
                    for i in xrange(repeats):
                        # Run the optimisation
                        uuid = str(attr_name) + str(i)
                        de = DE(problem, uuid, dimensions=d, verbosity=0, multiprocessing=False, **kwargs)
                        run_name = '- Run %s with %s = %s'%(i+1, var_name, attr)
                        try:
                            population, evals, generations = de.optimise()
                            print '%s:\tConverged with %s function evaluations over %s generations (Np = %s).'%(
                                run_name, evals, generations, population.size)
                        except NotConvergedException, best:
                            failures += 1
                            print '%s:\tFailure. Best value: %s'%(run_name, best[1])
                        # Dump convergence history
                        with open(uuid + '.csv', 'rb') as csvfile:
                            reader = csv.reader(csvfile)
                            for row_index, row in enumerate(reader):
                                for column_index, cell in enumerate(row):
                                    column_letter = get_column_letter((4 * i) + (column_index + 1))
                                    ws.cell('%s%s'%(column_letter, (row_index + 1))).value = cell
                        os.remove(uuid + '.csv')
                        del de
                        
                    
    print 'Writing results to Excel...'
    # Remove the default sheet
    ws = wb.get_sheet_by_name('Sheet')
    wb.remove_sheet(ws)
    wb.save(wb_name)


if __name__ == '__main__':
    tests()