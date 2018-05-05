# We rely on numpy for array/vector operations and advanced maths.
import numpy
import population

"""
Blake Hemingway, University of Sheffield, May 2014

Differential Evolution (DE) is an optimisation algorithm that minimises
an n-dimensional cost function using a population-based approach.

This file contains base classes representing generalised forms of the
most common DE variants.

More advanced forms of DE, with, for example, adaptive parameter controls,
may be implemented as subclasses of these.
Options (e.g. parallel cost function evaluation, logging to CSV file, different
termination criteria, crossover techniques etc.) can be implemented as mix-ins.
"""


class DERand1Bin(object):
    """
    This is 'classic' DE as outlined in Storn and Price's "Differential
    Evolution: A Practical Approach to Global Optimization".
    """

    def __init__(self, costFile, np=50, f=0.8, cr=0.9, maxFunctionEvals=50000):
        """
        This function is called when the DifferentialEvolution class is
        instantiated. A 'costFile' must be passed: this should be an included
        Python module containing the following methods:

        - cost(x): returning a scalar when passed a single vector argument x
        - getBounds(): returning a tuple, length 2, of the initialisation region.

        A Boolean 'absoluteBounds' may optionally be set. If this is True,
         mutations outside the initialisation region will be banned.

        Control parameters np (population size), f (mutation scaling factor) and
        cr (crossover ratio) can be specified, or left as the 'standard' values
        as stated by Qin & Suganthan (np) and Storn (cr, f).
        np will be, by default, conservatively large for low-dimensional problems.
        np = 10d, where d is problem dimensionality, is recommended in such cases.
        """
        # Get cost function
        self.cost = costFile.cost
        # Get problem boundaries
        self.minVector, self.maxVector = costFile.getBounds()
        # Infer the problem dimensionality from one of these (arbitrarily)
        self.d = len(self.minVector)
        # Check for 'phantom' indices, whereby the min and max vectors
        # constrain a variable to be constant.
        self.phantomIndices = []
        vectorDifference = numpy.array(self.maxVector) - numpy.array(self.minVector)
        for i, x in enumerate(vectorDifference):
            if x == 0:
                self.phantomIndices.append(i)
        # Initialise population randomly within the boundaries.
        self.population = population.Population(size=np, boundaries=(self.minVector, self.maxVector))
        # Are mutations outside these boundaries banned?
        try:
            self.absoluteBounds = costFile.absoluteBounds
        except AttributeError:
            self.absoluteBounds = False
        # Number of function evaluations before the program terminates
        self.maxFunctionEvals = maxFunctionEvals
        # The number of function evaluations now is, obviously, 0.
        self.functionEvaluations = 0
        # Define DE parameters
        self.cr = cr
        self.f = f

    def _nmeri(self, n, maximum, exclude=[]):
        """
        选择最大值为maximum的n个不重复随机数，除去exclude中的数字
        Helper function to return N Mutually Exclusive Random Integers (nmeri)
        in the range [0, maximum). Optionally takes an 'exclude'
        argument; a list of integers which will be excluded from the list.
        """
        selected = set()
        while len(selected) < n:
            rand = numpy.random.randint(maximum)
            # No need to check if rand in selected as selected is a set.
            if rand not in exclude:
                selected.add(rand)
        return list(selected)

    def mutation(self, i, f, n=1):
        """
        对第i个个体变异（DE/rand/n）
        The mutation style used by de/rand/n.
        Create a mutant individual by adding n scaled vector differences
        to a base vector in the main population.
        """
        r = self._nmeri(1 + 2*n, self.population.size, exclude=[i]) # 生成1+2*n（n=1时，为3）个，最大值为population.size，排除i的随机数
        baseVector = self.population.members[r.pop()].vector # x1
        while r: # 当r:list不为空
            v1, v2 = [self.population.members[j].vector for j in (r.pop(), r.pop())]
            try:
                difference += v1 - v2
            except NameError: # 如果报“变量没有先定义后使用”，则执行difference = v1 - v2
                difference = v1 - v2
        return population.Member(baseVector + f * difference)

    def crossover(self, parentIndex, mutant, cr):
        """
        Create a trial member by crossing a parent with a mutant.
        This function uses a binomial distribution to do so,
        in the style of DE/X/X/bin.
        The probability of a mutant element being selected over a parent element is
        cr, the crossover ratio. There also exists an 'iRand' to guarantee that
        at least one mutant value is chosen.
        """
        parent = self.population.members[parentIndex]
        # Exclude phantom indices as choices for iRand
        iRand = self._nmeri(1, self.d, exclude=self.phantomIndices)[0]
        for i in range(self.d):
            if numpy.random.rand() > cr and i != iRand:
                mutant.vector[i] = parent.vector[i]
        # 'mutant' is now not strictly a mutant but a trial member.
        return mutant

    def generateTrialMember(self, i):
        """
        生成第i个个体的经变异后重组的个体
        Generate a single trial member on parent index i
        by calling mutation and crossover operations.
        """
        mutant = self.mutation(i, self.f) # 获得第i个变异体
        trialMember = self.crossover(i, mutant, self.cr) # 由变异体获得重组后的第i个个体
        return trialMember

    def generateTrialPopulation(self, np):
        """
        生成重组后的种群，种群数量np
        Create a trial population (size np) from an existing population.
        Return a population object.
        """
        trialMembers = []
        for i in range(np):
            trialMember = self.generateTrialMember(i)
            # Bring the trial member back into the feasible region if necessary.
            if self.absoluteBounds: # absoluteBounds设置为True，则范围在限定范围外的个体将被“纠正”
                trialMember.constrain(self.minVector, self.maxVector)
            trialMembers.append(trialMember)
        return population.Population(members=trialMembers)

    def assignCosts(self, population):
        """
        为每个个体评分
        Compute and assign cost function values to each member of the passed
        population.Population instance by considering the member vectors.
        Return the modified population.
        """
        for i, member in enumerate(population.members):
            population.members[i].cost = self.cost(member.vector)
            self.functionEvaluations += 1
        return population

    def trialMemberSuccess(self, i, trialMember):
        """
        This function is called in the event of trialMember being found to be
        superior to its parent with index i in the population.
        Its main action is to replace the losing population member
        with the victorious trial member.
        """
        self.population.members[i] = trialMember

    def trialMemberFailure(self, i, trialMember):
        """
        This function is called in the event of trialMember being found to be
        inferior to its parent with index i in the population.
        By default, it does nothing, but provides a 'hook' for future extensibility.
        """
        pass

    def selectNextGeneration(self, trialPopulation):
        """
        通过评分，选择进入下一代的个体，进而获得进化后的种群，trialPopulation是变异重组后的种群
        Compare the main population with the trial population by cost.
        """
        for i, trialMember in enumerate(trialPopulation.members):
            # <= (not <) is important to avoid stagnation in quantised landscapes.
            if trialMember.cost <= self.population.members[i].cost:
                self.trialMemberSuccess(i, trialMember) # 第i个变异重组的个体，成功进入下一代
            else:
                self.trialMemberFailure(i, trialMember) # 第i个变异重组的个体没有进入下一代

    def terminationCriterion(self):
        """
        停机条件
        Termination is based on a limited number of function evaluations.
        """
        return self.functionEvaluations >= self.maxFunctionEvals

    def optimise(self):
        """
        DE主方法
        The main method. Call this method to run the optimisation.
        """
        self.population = self.assignCosts(self.population)
        # A generation counter is provided for future extensibility,
        # but is not used by basic DE.
        self.generation = 0
        while self.terminationCriterion() == False:
            self.generation += 1
            # Generate (mutate/crossover) a trial population
            trialPopulation = self.generateTrialPopulation(self.population.size) # 生成变异重组后的个体
            # Evaluate the trial population
            trialPopulation = self.assignCosts(trialPopulation) # 为变异重组后的种群评分，分数由population.members[i].cost = self.cost(member.vector)形式带回
            # Insert improvements
            self.selectNextGeneration(trialPopulation) # 生成新种群
        return self.population.bestVector


class DECurrentToPBest1Bin(DERand1Bin):
    """
    此类根据k和p，概括了多个变异策略
    Constructs mutants in accordance with the following procedure:

    u_i = x_i + k*(x_pbest - x_i) + f_i*(x_r1 - x_r2)

    Where x_pbest is randomly chosen as one of the top 100 p% individuals,

    The following algorithms are specific cases:

    - DE/best/1/bin (k = 1, p = 0)
    - DE/current-to-best/1/bin (k = fi, p = 0)
    - DE/current-to-rand/1/bin (k = fi, p = 1)

    k is taken as equal to fi unless otherwise specified in the constructor.
    k may also be callable (e.g. a random number generating function),
    in which case it will be evaluated without arguments.
    p = 0.05 by default, as given in Zhang and Sanderson's JADE.
    """

    def mutation(self, i, f, n=1, k=None, p=0.05):
        '''
        对第i个个体变异
        :param i: 个体序号
        :param f: F值，变异系数
        :param n: 参与变异的平庸个体的个数*2
        :param k: 根据k和p可以对应多种变异策略，参见DECurrentToPBest1Bin初始化函数
        :param p:
        :return:
        '''
        if k is None:
            k = f
        else:
            try:
                k = k()
            except TypeError:
                pass
        # Obtain one of the p-best vectors at random (population is sorted by cost)
        # 获取以cost（评分）排序的最好的p*100%个个体中随机的一个,,,似乎将最好的个体排在最前面
        if p:
            maxPBestIndex = numpy.ceil(p * self.population.size)
            pBestIndex = numpy.random.randint(maxPBestIndex)
        else:
            pBestIndex = 0
        pBestVector = self.population.members[pBestIndex].vector
        # Obtain the current (aka target) vector
        currentVector = self.population.members[i].vector
        # Compute the base vector
        baseVector = currentVector + k * (pBestVector - currentVector)
        # Compute a difference vector
        r = self._nmeri(2*n, self.population.size, exclude=[i, pBestIndex])
        while r:
            v1, v2 = [self.population.members[j].vector for j in (r.pop(), r.pop())]
            try:
                difference += v1 - v2
            except NameError:
                difference = v1 - v2
        return population.Member(baseVector + f * difference)

    def generateTrialPopulation(self, *args, **kwargs):
        """
        此处将种群中的个体按照cost排序
        Sort main population by cost before generating trial population.
        """
        self.population.members.sort(key=lambda x: x.cost)
        return super(DECurrentToPBest1Bin, self).generateTrialPopulation(*args, **kwargs)
