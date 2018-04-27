# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 16:08
# @Author  : MengnanChen
# @FileName: DE_RL.py
# @Software: PyCharm Community Edition

import population
from deBase import DECurrentToPBest1Bin,DERand1Bin
import numpy as np
from collections import deque
import pandas as pd
import itertools
from copy import deepcopy
import time
import os

global INF
INF=float('inf') # 无穷大的数

DEBUG=True
DEBUG_FILE=True
tmp_id=str(time.strftime('%dd_%mmon_%Y_%Hh_%Mm'))
tmp_dir='tmp_'+tmp_id
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

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
    def __init__(self, *args, **kwargs):
        kwargs['f'] = 0.5
        kwargs['cr'] = 0.5
        super(rlde,self).__init__(*args,**kwargs)

        self.nm=0 # 最小适应值停滞观测变量
        self.rm=0 # 最小适应值持续减小代数观测变量
        self.um=0 # 种群数量维持在种群上界观测变量
        self.lm=0 # 种群数量维持在种群下界观测变量
        self.mean_s=0 # 平均适应值停滞
        self.mean_cd=0 # 平均适应值持续下降
        self.std_s=0 # 方差适应值停滞
        self.std_cd=0 # 方差适应值持续下降
        self.state_prior=self.current_state=(self.mean_s,self.mean_cd,self.um,self.lm,self.std_s,self.std_cd) # 之前一次的state

        self.min_fitness=INF # 当前适应值最小值
        self.mean_fitness=INF # 当前适应值的平均值
        self.std_fitness=INF # 当前适应值的方差

        self.Ubound=200 # population_size 的上界
        self.Lbound=50 # population_size 的下界

        self.p=0.1 # 种群增减策略中没测选取的最优/最差候选个体占总个体的比例
        self.change_p=0.05 # 每次执行增减种群的占总个体的比例

        self.nm_weight=-2 # 奖赏中，各种状态的权重，停滞观测变量
        self.rm_weight=-0.5 # 冗余观测器变量
        self.mean_s_weight=-10 # 平均适应值停滞权重
        self.mean_cd_weight=-1 # 平均适应值持续下降权重
        self.std_s_weight=-10 # 方差适应值停滞
        self.std_cd_weight=-1 #方差适应值持续下降
        self.um_weight=-1 # 种群数量维持上界观测器变量
        self.lm_weight=-10 # 种群数量维持下界观测器变量
        self.fitness_decay_weight=1e-8 # 适应值降低奖赏
        self.actions_count=9 # 动作个数
        self.actions=list(range(self.actions_count))
        self.lr=0.01 # 学习率
        self.gamma=0.9 # 奖励衰减，使得算法能够看清大局
        self.epsilon=0.9 # 贪婪度
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64) # 初始 q_table
        self.mean_s_threadhold=5e7 # 自适应
        self.std_s_threadhold=1e9 # 自适用
        self.mean_s_threadhold_q=deque(maxlen=3)
        self.std_s_threadhold_q=deque(maxlen=5)

        self.evolve_policy=0
        self.action=np.random.choice(self.actions)

        self.lp=50
        # 类似SaDE, 仿照SaDE
        self.strategies=[
            {
                'algorithm': DERand1Bin,
                'probability': 0.25,
                'cr': 0.5,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToBest2Bin,
                'probability': 0.25,
                'cr': 0.5,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DERand2Bin,
                'probability': 0.25,
                'cr': 0.5,
                'crMemory': deque(maxlen=self.lp)
            },
            {
                'algorithm': DECurrentToRand1,
                'probability': 0.25,
                'cr': 0.5,
                'crMemory': deque(maxlen=self.lp)
            },
        ]

    def update_state(self):  # 更新观测器(self.mean_s,self.mean_cd,self.um,self.lm,self.std_s,self.std_cd)
        if self.mean_fitness - np.mean(self.population.costs) > self.mean_s_threadhold:
            if self.mean_cd<20:
                self.mean_cd += 1
            self.mean_s = 0
        else:
            self.mean_cd = 0
            if self.mean_s<20:
                self.mean_s += 1
        self.mean_fitness = np.mean(self.population.costs)
        self.mean_s_threadhold_q.append(self.mean_fitness - np.mean(self.population.costs))
        if self.population.size >= self.Ubound:
            if self.um<20:
                self.um += 1
        elif self.population.size <= self.Lbound:
            if self.lm<20:
                self.lm += 1
            self.um = 0
        if self.std_fitness - np.std(self.population.costs) > self.std_s_threadhold:
            if self.std_cd <20:
                self.std_cd += 1
            self.std_s = 0
        else:
            if self.std_s <20:
                self.std_s += 1
            self.std_cd = 0
        self.std_fitness = np.std(self.population.costs)
        self.std_s_threadhold_q.append(self.std_fitness - np.std(self.population.costs))
        # 使mean_s_threadhold 和 std_s_threadhold 自适应
        if len(self.mean_s_threadhold_q) >= 3:
            self.mean_s_threadhold = np.mean(self.mean_s_threadhold_q)
        if len(self.std_s_threadhold_q) >= 5:
            self.std_s_threadhold = np.mean(self.std_s_threadhold_q)

    def add_individual(self):
        if not (self.population.size < self.Ubound and self.population.size > self.Lbound):  # 当self.population_size 小于种群上界且大于种群上界
            return
        max_p_best_index=np.ceil(self.p*self.population.size)
        best_vectors=[] # 是一个Member对象的list
        p_best_index=np.random.permutation(int(max_p_best_index))
        change_size=int(np.ceil(self.change_p*self.population.size))
        for i in range(change_size):
            p_best = deepcopy(self.population.members[p_best_index[i]])
            for index,ele in enumerate(p_best.vector): # 轻微扰动
                p_best.vector[index]+=np.random.uniform(-1.0/pow(1.001,self.functionEvaluations),1.0/pow(1.001,self.functionEvaluations))
            best_vectors.append(p_best)
        self.population.members.extend(best_vectors) # population.members是一个Member对象的list

    def substract_individual(self):
        if not (self.population.size < self.Ubound and self.population.size > self.Lbound):  # 当self.population_size 小于种群上界且大于种群上界
            return
        max_p_worst_index=np.ceil(self.p*self.population.size)
        p_worst_index=np.random.permutation(int(max_p_worst_index))
        p_worst_index=[x for x in p_worst_index if x!=0]
        change_size=int(np.ceil(self.change_p*self.population.size))
        for i in range(change_size):
            p_worst=self.population.members[-p_worst_index[i]]
            self.population.members.remove(p_worst)

    def add_f(self):
        if self.f<2.0:
            self.f+=0.1

    def substract_f(self):
        if self.f>0:
            self.f-=0.1

    def update_cr(self):
        for index,strategy in enumerate(self.strategies):
            flattendedCr=list(itertools.chain.from_iterable(strategy['crMemory']))
            if flattendedCr:
                self.strategies[index]['cr']=np.median(flattendedCr)

    def update_to_policy1(self):
        self.evolve_policy=0
    def update_to_policy2(self):
        self.evolve_policy=1
    def update_to_policy3(self):
        self.evolve_policy=2
    def update_to_policy4(self):
        self.evolve_policy=3

    def choose_action(self,state):
        self.check_state_exist(state)
        if np.random.uniform()<self.epsilon:
            state_action=self.q_table.loc[state,:]
            state_action=state_action.reindex(np.random.permutation(state_action.index)) #重置dataframe的索引（索引随机分配）
            self.action=state_action.idxmax()
        else:
            self.action=np.random.choice(self.actions)
        return self.action

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table=self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def update_population(self):
        # state=(self.nm,sedlf.rm,self.um,self.lm) # 第t次
        self.current_state=(self.mean_s,self.mean_cd,self.um,self.lm,self.std_s,self.std_cd)
        action=self.choose_action(str(self.state_prior))

        if action==0: # 增加种群
            self.add_individual()
        elif action==1: # 减少种群
            self.substract_individual()
        elif action==2: # 更新变异率CR正态分布中值
            self.update_cr()
        elif action==3: # 增加F值正态分布中值
            self.add_f()
        elif action==4: # 减小F值正态分布中值
            self.substract_f()
        elif action==5: # 更新变异策略至1
            self.update_to_policy1()
        elif action==6: # 更新变异策略至2
            self.update_to_policy2()
        elif action==7: # 更新变异策略至3
            self.update_to_policy3()
        elif action==8: # 更新变异策略至4
            self.update_to_policy4()

    def update_q_table(self,prior_score,current_score):
        # 将由于fitness_decay_weight 获得的奖励控制在0~100
        digits=1
        while abs(current_score-prior_score)/(pow(10,digits))>100:
            digits+=1
        self.fitness_decay_weight=1.0/pow(10,digits)
        reward = self.mean_s_weight* self.mean_s + self.mean_cd_weight * self.mean_cd + self.um * self.um_weight + self.lm * self.lm_weight \
                 +self.std_s_weight*self.std_s+self.std_cd_weight*self.std_cd+ self.fitness_decay_weight * (prior_score - current_score)
        self.check_state_exist(str(self.current_state))
        q_predict=self.q_table.loc[str(self.state_prior),self.action]
        q_target=reward+self.gamma*self.q_table.loc[str(self.current_state),:].max()
        self.q_table.loc[str(self.state_prior),self.action]+=self.lr*(q_target-q_predict)
        self.state_prior=self.current_state

    def generateTrialMember(self, i):
        strategy=self.strategies[self.evolve_policy]
        if self.f is None:
            use_f=np.random.normal(0.5,0.3)
            self.f=use_f
        else:
            use_f=np.random.normal(self.f,0.3)
        while True:
            cri=np.random.normal(strategy['cr'],0.1)
            if cri>=0 and cri<=1:
                break
        mutant=strategy['algorithm'].mutation(self,i,use_f)
        trialMember=strategy['algorithm'].crossover(self,i,mutant,cri)
        trialMember.strategy=self.evolve_policy
        trialMember.cr=cri
        return trialMember

    def generateTrialPopulation(self, *args, **kwargs):
        n=len(self.strategies)
        for i in range(n):
            self.strategies[i]['crMemory'].append([])
        self.prior_scores=deepcopy(self.population.costs)
        return super(rlde, self).generateTrialPopulation(*args, **kwargs)

    def trialMemberSuccess(self, i, trialMember):
        self.strategies[trialMember.strategy]['crMemory'][-1].append(trialMember.cr)
        super(rlde,self).trialMemberSuccess(i,trialMember)

    def optimise(self):
        self.population = self.assignCosts(self.population)
        self.generation = 0
        while self.terminationCriterion() == False:
            self.generation += 1
            trialPopulation = self.generateTrialPopulation(self.population.size) # 生成变异重组后的个体
            trialPopulation = self.assignCosts(trialPopulation) # 为变异重组后的种群评分，分数由population.members[i].cost = self.cost(member.vector)形式带回

            if DEBUG:
                if DEBUG_FILE:
                    q_table_id=time.strftime('%dd_%mmon_%Y_%Hh_%Mm')
                    with open(tmp_dir+'/q_table_'+q_table_id, 'a+') as f:
                        print(f'FEs:{self.functionEvaluations}:', self.q_table,file=f)
                else:
                    print(f'FEs:-{self.functionEvaluations}:', self.q_table)

            self.update_q_table(np.mean(self.prior_scores),np.mean(trialPopulation.costs))
            self.update_state()

            self.selectNextGeneration(trialPopulation) # 生成新种群

            self.update_population()
        return self.population.bestVector