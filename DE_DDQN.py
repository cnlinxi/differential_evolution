#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 22:01
# @Author  : MengnanChen
# @Site    :
# @File    : DE_DDQN.py
# @Software: PyCharm Community Edition

import population
from deBase import DECurrentToPBest1Bin,DERand1Bin
import numpy as np
import tensorflow as tf
from collections import deque
import pandas as pd
import itertools
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from DDQN import DQNAgent

np.random.seed(1)
tf.set_random_seed(1)

class DECurrentToBest2Bin(DECurrentToPBest1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['p'] = 0
        kwargs['n'] = 2
        return DECurrentToPBest1Bin.mutation(self, *args, **kwargs)


class DERand2Bin(DERand1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['n'] = 2
        return DERand1Bin.mutation(self, *args, **kwargs)


class DECurrentToRand1(DECurrentToPBest1Bin):
    def mutation(self, *args, **kwargs):
        kwargs['p'] = 1
        return DECurrentToPBest1Bin.mutation(self, *args, **kwargs)

    def crossover(self, parentIndex, mutant, cr):
        """
        This algorithm does not implement crossover to retain rotational
        invariance.
        """
        return mutant

class rlde(DECurrentToPBest1Bin):
    def __init__(self,*args, **kwargs):
        kwargs['f'] = 0.5
        kwargs['cr'] = 0.5
        super(rlde,self).__init__(*args,**kwargs)

        self.lp=50
        # 类似SaDE, 仿照SaDE
        self.strategies=[
            {
                'algorithm': DERand1Bin,
                'f': 1.0,
                'f_min':0.1,
                'f_max':1.0,
                'fMemory':deque(maxlen=self.lp),
                'cr': 0.8,
                'cr_min':0.1,
                'cr_max':1.0,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToBest2Bin,
                'f': 1.0,
                'f_min': 0.1,
                'f_max': 1.0,
                'fMemory':deque(maxlen=self.lp),
                'cr': 0.8,
                'cr_min': 0.1,
                'cr_max': 1.0,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DERand2Bin,
                'f': 1.0,
                'f_min': 0.1,
                'f_max': 1.0,
                'fMemory':deque(maxlen=self.lp),
                'cr': 0.8,
                'cr_min': 0.1,
                'cr_max': 1.0,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToRand1,
                'f': 1.0,
                'f_min': 0.1,
                'f_max': 1.0,
                'fMemory':deque(maxlen=self.lp),
                'cr': 0.8,
                'cr_min': 0.1,
                'cr_max': 1.0,
                'crMemory': deque(maxlen=self.lp)
            },
        ]

        self.p=0.1
        self.change_p=0.05

        self.evolve_policy=0 # 保存当前的变异策略
        self.n_actions=4 # 动作个数
        self.actions=list(range(self.n_actions))
        self.action=np.random.choice(self.actions)
        self.dqn=DQNAgent(state_size=self.d,action_size=self.n_actions) # self.d 个体向量维数

    def update_cr(self):
        for index,strategy in enumerate(self.strategies):
            flattendedCr=list(itertools.chain.from_iterable(strategy['crMemory']))
            if flattendedCr:
                self.strategies[index]['cr']=np.median(flattendedCr)
                self.strategies[index]['cr_min']=np.min(flattendedCr)
                self.strategies[index]['cr_max']=np.max(flattendedCr)

    def update_f(self):
        for index,strategy in enumerate(self.strategies):
            flattendedF=list(itertools.chain.from_iterable(strategy['fMemory']))
            if flattendedF:
                self.strategies[index]['f']=np.median(flattendedF)
                self.strategies[index]['f_min']=np.min(flattendedF)
                self.strategies[index]['f_max']=np.max(flattendedF)

    def update_population(self,action):
        if action==0:
            self.evolve_policy = 0
        elif action==1:
            self.evolve_policy = 1
        elif action==2:
            self.evolve_policy = 2
        elif action==3:
            self.evolve_policy = 3

    def assign_cost(self,member):
        member.cost = self.cost(member.vector)
        self.functionEvaluations += 1
        self.current_trial_costs.append(member.cost)
        return member

    def assign_population_costs(self,population):
        for i, member in enumerate(population.members):
            population.members[i].cost = self.current_trial_costs[i]
        return population

    def generateTrialMember(self, i):
        member_i=self.population.members[i]
        self.action=self.dqn.choose_action(member_i.vector)
        self.update_population(self.action)
        strategy=self.strategies[self.evolve_policy]
        use_f=np.median((strategy['f_min'],strategy['f_max'],strategy['f']+ 0.1 * np.random.standard_cauchy()))
        use_cr=np.median((strategy['cr_min'],strategy['cr_max'],np.random.normal(strategy['cr'],0.3)))
        mutant=strategy['algorithm'].mutation(self,i,use_f)
        trialMember=strategy['algorithm'].crossover(self,i,mutant,use_cr)
        trialMember.strategy=self.evolve_policy
        trialMember.cr=use_cr
        trialMember.f=use_f
        return trialMember

    def trialMemberSuccess(self, i, trialMember):
        self.strategies[trialMember.strategy]['crMemory'][-1].append(trialMember.cr)
        self.strategies[trialMember.strategy]['fMemory'][-1].append(trialMember.f)
        self.update_f()
        self.update_cr()
        self.n_member_success+=1
        super(rlde,self).trialMemberSuccess(i,trialMember)

    def update_population_size(self):
        self.population.members.sort(key=lambda x: x.cost)
        if float(self.n_member_success)/self.population.size>0.5:
            self.add_individual()
        else:
            self.substract_individual()

    def generateTrialPopulation(self, np):
        n=len(self.strategies)
        for i in range(n):
            self.strategies[i]['crMemory'].append([])
        for i in range(n):
            self.strategies[i]['fMemory'].append([])
        self.prior_scores=deepcopy(self.population.costs)
        trialMembers = []
        for i in range(np):
            trialMember = self.generateTrialMember(i)
            if self.absoluteBounds: # absoluteBounds设置为True，则范围在限定范围外的个体将被“纠正”
                trialMember.constrain(self.minVector, self.maxVector)
            trial_member = self.assign_cost(trialMember)
            trialMembers.append(trialMember)
            reward=-100
            if self.population.members[i].cost - trial_member.cost>0: #上次的cost比本次的cost大，正向奖励
                reward=10
            self.dqn.store_transition(self.population.members[i].vector,self.action,reward,trialMember.vector)
            self.dqn.learn() # learning
        return population.Population(members=trialMembers)

    def substract_individual(self):
        try:
            if not self.population.size >= self.d:
                return
            max_p_worst_index = np.ceil(self.p * self.population.size)
            p_worst_index = np.random.permutation(int(max_p_worst_index))
            p_worst_index = [x for x in p_worst_index if x != 0]
            if len(p_worst_index)<1:
                return
            change_size = int(np.ceil(self.change_p * self.population.size))
            for i in range(change_size):
                p_worst = self.population.members[-p_worst_index[i]]
                self.population.members.remove(p_worst)
        except:
            print('substract individual error')
            pass

    def add_individual(self):
        try:
            max_p_best_index = np.ceil(self.p * self.population.size)
            best_vectors = []  # 是一个Member对象的list
            p_best_index = np.random.permutation(int(max_p_best_index))
            change_size = int(np.ceil(self.change_p * self.population.size))
            for i in range(change_size):
                p_best = deepcopy(self.population.members[p_best_index[i]])
                for index, ele in enumerate(p_best.vector):  # 轻微扰动
                    p_best.vector[index] += np.random.uniform(-1.0 / pow(1.01, self.functionEvaluations),
                                                              1.0 / pow(1.01, self.functionEvaluations))
                best_vectors.append(p_best)
            self.population.members.extend(best_vectors)  # population.members是一个Member对象的list
        except:
            print('add individual error')
            pass

    def optimise(self):
        self.population = self.assignCosts(self.population)
        self.generation = 0
        while self.terminationCriterion() == False:
            self.generation += 1
            self.current_trial_members=[]
            self.current_trial_costs=[]
            self.n_member_success=0
            trialPopulation = self.generateTrialPopulation(self.population.size) # 生成变异重组后的个体
            trialPopulation = self.assign_population_costs(trialPopulation) # 为变异重组后的种群评分，分数由population.members[i].cost = self.cost(member.vector)形式带回
            self.selectNextGeneration(trialPopulation) # 生成新种群
<<<<<<< HEAD
            # self.update_population_size()
=======
            self.update_population_size()
>>>>>>> 3bcb3db40eaef5b03cad30f85105df93bf417552
            print(f'Evaluation:{self.functionEvaluations}, Current Cost:{np.mean(self.population.costs)}, Population size:{self.population.size}')
        return self.population.bestVector