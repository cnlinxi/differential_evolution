import numpy, openpyxl, string, test_functions
from differential_evolution import DifferentialEvolution, NotConvergedException, SelfAdaptiveDifferentialEvolution
ALPHABET = string.uppercase

'''
This file tests the various options of the DifferentialEvolution class by
running them repeatedly on the test functions defined in test_functions.py.

The results are exported to Microsoft Excel.
'''

class TestDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
    '''
    Modifications to the standard DE class to make it appropriate for tests.
    '''
    def __init__(self, d):
        self.dimensionality = d
        super(TestDifferentialEvolution, self).__init__()
        self.verbosity = 0
        self.convergence_function = 'vtr'
        self.mutation_scheme = 'de/rand/1/bin'
        self.base_vector_selection_scheme = 'permuted'
        self.population_size = int(11.69 * self.dimensionality**0.63)  # Power law obtained from Np = 50 when d = 10, Np = 100 when d = 30
        self.f = 0.8
        self.f_randomisation = 'static'
        self.f_decay = False
        self.c = 0.5
        self.c_randomisation = 'dither'
        self.max_generations = 100 * self.population_size


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
    dimensions = [3, 5, 10]
    variables = {
        #'population_size': [10, 30, 100],
        'f': [0.3, 0.5, 0.7, 0.9],
        #'c': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # 'mutation_scheme': ['de/rand/1/bin', 'de/best/1/bin', 'de/current_to_best/1/bin', 'de/rand/2/bin', 'de/best/2/bin', 'de/rand_then_best/1/bin'],
        #'base_vector_selection_scheme': ['random', 'permuted', 'offset'],
        #'f_randomisation': ['static', 'dither', 'jitter'],
        #'c_randomisation': ['dither'],
    }
    repeats = 10
    # Initialise Excel workbook
    wb = openpyxl.Workbook()
    wb_name = 'DE_Tests.xlsx'
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
            'sphere': test_functions.SphereDifferentialEvolution(d=d),
            'hyper-ellipsoid': test_functions.HyperEllipsoidDifferentialEvolution(d=d),
            'rozenbrock': test_functions.RozenbrockDifferentialEvolution(d=d),
            'schwefel-ridge': test_functions.SchwefelRidgeDifferentialEvolution(d=d),
            'neumaier': test_functions.NeumaierDifferentialEvolution(d=d), # NONZERO TARGET
        }
        multimodal_problems = {
            #'ackley': test_functions.AckleyDifferentialEvolution(d=d),
            #'griewangk': test_functions.GriewangkDifferentialEvolution(d=d),
            #'rastrigin': test_functions.RastriginDifferentialEvolution(d=d),
            #'salomon': test_functions.SalomonDifferentialEvolution(d=d),
            #'whitley': test_functions.WhitleyDifferentialEvolution(d=d),
            #'storn': test_functions.StornDifferentialEvolution(d=d),
            #'lennard-jones': test_functions.LennardJonesDifferentialEvolution(d=d),
            #'hilbert': test_functions.HilbertDifferentialEvolution(d=d),
            #'modified-langerman': test_functions.ModifiedLangermanDifferentialEvolution(d=d),
            #'shekel': test_functions.ShekelDifferentialEvolution(d=d),
            #'odd-square': test_functions.OddSquareDifferentialEvolution(d=d),
            #'katsuura': test_functions.KatsuuraDifferentialEvolution(d=d),
        }
        bound_problems = {
            #'schwefel': test_functions.SchwefelDifferentialEvolution(d=d), # NONZERO TARGET
            #'michalewicz': test_functions.MichalewiczDifferentialEvolution(d=d),
            #'rana': test_functions.RanaDifferentialEvolution(d=d)  # NONZERO TARGET
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
                    for i in xrange(repeats):
                        setattr(problem, var_name, attr)
                        run_name = '- Run %s with %s = %s'%(i+1, var_name, attr)
                        try:
                            solution, generations, evals, np, c, f_thresholds, m_thresholds = problem.solve()
                            generations_log.append(generations)
                            function_evaluations_log.append(evals)
                            print '%s:\tConverged with %s function evaluations over %s generations (Np = %s). c=%s, f=%s, m=%s'%(
                                run_name, evals, generations, np, c, f_thresholds, m_thresholds)
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