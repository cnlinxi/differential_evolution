import numpy, openpyxl, string, testFunctions
from differentialEvolution import DifferentialEvolution2 as DE, NotConvergedException
ALPHABET = string.uppercase
import time

'''
This file tests the various options of the DifferentialEvolution class by
running them repeatedly on the test functions defined in testFunctions folder.

The results are exported to Microsoft Excel.
'''

def initialise_worksheet(ws, var_name, var_list):
    '''
    Adds appropriate headers and formatting to an
    openpyxl worksheet as used in DE testing.
    '''
    ws.cell('A1').value = 'Function'
    ws.cell('B1').value = 'Dimensions'
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 15
    for i, var in enumerate(var_list):
        cells_string = '%s1:%s1'%(ALPHABET[2 + i*2], ALPHABET[3 + i*2])
        ws.merge_cells(cells_string)
        ws.cell(row=0, column=(2 + i*2)).value = '%s = %s'%(var_name, var)
        ws.cell(row=0, column=(2 + i*2)).style.alignment.horizontal = 'center'
        ws.cell(row=1, column=(2 + i*2)).value = 'Success Ratio'
        ws.cell(row=1, column=(3 + i*2)).value = 'Function Evals'
        ws.column_dimensions[ALPHABET[2 + i*2]].width = 15
        ws.column_dimensions[ALPHABET[3 + i*2]].width = 15
    # Make first two rows bold
    max_col = ALPHABET[2 + 2*len(var_list)]
    for row in ws.range('A1:%s2'%(max_col)):
        for cell in row:
            cell.style.font.bold = True
    return ws


def tests():
    dimensions = [5, 10]
    variables = {
        #'populationSize': [10],
        #'f': [0.3, 0.6, 0.9],
        'cr': [0.9, None],
        #'mutator': ['de/rand/1/bin', 'de/best/1/bin', 'de/current_to_best/1/bin', 'de/rand/2/bin', 'de/best/2/bin', 'de/rand_then_best/1/bin'],
        #'baseVectorSelector': ['random', 'permuted', 'offset'],
        #'fRandomiser': ['static', 'dither', 'jitter', 'randomInterval'],
        #'huddling': [True, False]
    }
    repeats = 5
    # Initialise Excel workbook
    wb = openpyxl.Workbook()
    wb_name = 'DE_Tests_%s.xlsx'%(time.strftime('%d-%m-%Y__%H:%M'))
    worksheets = {}
    for var_name, var_list in variables.iteritems():
        worksheets[var_name] = wb.create_sheet()
        worksheets[var_name].title = var_name
        worksheets[var_name] = initialise_worksheet(worksheets[var_name],
            var_name, var_list)
    # Remove the default sheet
    ws = wb.get_sheet_by_name('Sheet')
    wb.remove_sheet(ws)
    # Run the tests
    for d in dimensions:
        unimodal_problems = {
            #'sphere': testFunctions.sphere,
            #'hyper-ellipsoid': testFunctions.hyperellipsoid,
            'rozenbrock': testFunctions.rozenbrock,
            #'schwefel-ridge': testFunctions.schwefelridge,
            #'neumaier': testFunctions.neumaier, # NONZERO TARGET
        }
        multimodal_problems = {
            #'ackley': testFunctions.ackley,
            #'griewangk': testFunctions.griewangk,
            #'rastrigin': testFunctions.rastrigin,
            #'salomon': testFunctions.salomon,
            #'whitley': testFunctions.whitley,
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
                results = [problem_name, d]
                for attr in var_list:
                    generations_log = []
                    function_evaluations_log = []
                    failures = 0
                    kwargs = {var_name: attr}
                    for i in xrange(repeats):
                        de = DE(problem, dimensions=d, verbosity=0, huddling=True, **kwargs)
                        run_name = '- Run %s with %s = %s'%(i+1, var_name, attr)
                        try:
                            population, evals, generations = de.optimise()
                            generations_log.append(generations)
                            function_evaluations_log.append(evals)
                            print '%s:\tConverged with %s function evaluations over %s generations (Np = %s).'%(
                                run_name, evals, generations, population.size)
                        except NotConvergedException, best:
                            failures += 1
                            print '%s:\tFailure. Best value: %s'%(run_name, best[1])
                    success_ratio = float(repeats - failures) / float(repeats)
                    results.append(numpy.around(success_ratio,3))
                    if failures != repeats:
                        average_fn_evals = numpy.mean(function_evaluations_log)
                        results.append(numpy.around(average_fn_evals,0))
                    else:
                        results.append('n/a')
                worksheets[var_name].append(results)
    print 'Writing results to Excel...'
    wb.save(wb_name)


if __name__ == '__main__':
    tests()