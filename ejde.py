# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 15:20
# @Author  : MengnanChen
# @FileName: ejde.py
# @Software: PyCharm Community Edition

from jde import jDE
import numpy as np
from population import Member
from copy import deepcopy
from random import choice


class ejDE(jDE):
    """
    The original jDE by Brest et al., using one strategy
    (DE/rand/1/bin).
    """
    def __init__(self, *args, **kwargs):
        super(ejDE, self).__init__(*args, **kwargs)
        for i in range(self.population.size):
            self.population.members[i].f = 0.1 + 0.9 * np.random.rand()
            self.population.members[i].cr = np.random.rand()

        self.p=0.2
        self.change_p=0.1
        self.f_best=0.1
        self.threadhold=1.0

    def update_population(self):
        # remove high cost solution
        change_size=int(np.ceil(self.change_p*self.population.size))
        max_p_worst_index=np.ceil(self.p*self.population.size)
        p_worst_index=np.random.permutation(int(max_p_worst_index)+1) # +1 是为了去除0之后少了那个数
        p_worst_index=[x for x in p_worst_index if x!=0]
        for i in range(change_size):
            p_worst=self.population.members[-p_worst_index[i]]
            self.population.members.remove(p_worst)

        # add low cost solution
        max_p_best_index=np.ceil(self.p*self.population.size)
        for i in range(change_size):
            p_best_index = np.random.permutation(int(max_p_best_index))
            p_best1=self.population.members[p_best_index[0]]
            p_best2=self.population.members[p_best_index[1]]
            p_best3=self.population.members[p_best_index[2]]
            best_mutation=deepcopy(p_best1)
            best_mutation.vector=p_best1.vector+self.f_best*(p_best2.vector-p_best3.vector)
            self.population.members.append(best_mutation)

    def optimise(self):
        self.population = self.assignCosts(self.population)
        self.population.members.sort(key=lambda x: x.cost) # sort the solution by fitness when initialization
        self.generation=0
        while self.terminationCriterion() == False:
            self.generation+=1
            trialPopulation = self.generateTrialPopulation(self.population.size)
            trialPopulation = self.assignCosts(trialPopulation)
            self.selectNextGeneration(trialPopulation)
            self.mean_std=np.mean(np.std(self.population.vectors,axis=0,ddof=1))
            # print(f'generation:{self.generation},len std:{len(np.std(self.population.vectors,axis=0,ddof=1))}, mean_std:{self.mean_std}')
            if self.mean_std<self.threadhold:
                self.update_population()
        return self.population.bestVector
