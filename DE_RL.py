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
import math

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
    def __init__(self,*args,*kwargs):
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

        self.Ubound=15000 # population_size 的上界
        self.Lbound=900 # population_size 的下界

        self.p=0.05 # 种群缩减策略中，去除的个体占总体的比例
        self.change_size=10 # 每次执行增加或减少种群的个体数

        self.mean_s_weight=-1 # 平均适应值停滞权重
        self.mean_cd_weight=-1 # 平均适应值持续下降权重
        self.std_s_weight=-1 # 方差适应值停滞
        self.std_cd_weight=-1 #方差适应值持续下降
        self.nm_weight=-2 # 奖赏中，各种状态的权重，停滞观测变量
        self.rm_weight=-0.5 # 冗余观测器变量
        self.um_weight=-1 # 种群数量维持上界观测器变量
        self.lm_weight=-1 # 种群数量维持下界观测器变量
        self.fitness_decay_weight=1e-8 # 适应值降低奖赏
        self.actions_count=9 # 动作个数
        self.actions=list(range(self.actions_count))
        self.action=0
        self.lr=0.01 # 学习率
        self.gamma=0.9 # 奖励衰减，使得算法能够看清大局
        self.epsilon=0.9 # 贪婪度
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64) # 初始 q_table
        self.mean_s_threadhold=5e7 # 自适应
        self.std_s_threadhold=1e9 # 自适用
        self.mean_s_threadhold_q=deque(maxlen=5)
        self.std_s_threadhold_q=deque(maxlen=5)
        # 类似SaDE, 仿照SaDE
        self.strategies=[
            [
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
        ]
        self.lp=50

    def update_state(self):  # 更新观测器(self.mean_s,self.mean_cd,self.um,self.lm,self.std_s,self.std_cd)
        print('update state mean:', self.mean_fitness - np.mean(self.scores))
        if self.mean_fitness - np.mean(self.scores) > self.mean_s_threadhold:
            self.mean_cd += 1
            self.mean_s = 0
        else:
            self.mean_cd = 0
            self.mean_s += 1
        self.mean_fitness = np.mean(self.scores)
        self.mean_s_threadhold_q.append(self.mean_fitness - np.mean(self.scores))
        if self.population.size >= self.Ubound:
            self.um += 1
            self.lm = 0
        elif self.population.size <= self.Lbound:
            self.lm += 1
            self.um = 0
        print('update state std:', self.std_fitness - np.std(self.scores))
        if self.std_fitness - np.std(self.scores) > self.std_s_threadhold:
            self.std_cd += 1
            self.std_s = 0
        else:
            self.std_cd = 0
            self.std_s += 1
        self.std_fitness = np.std(self.scores)
        self.std_s_threadhold_q.append(self.std_fitness - np.std(self.scores))
        # 使mean_s_threadhold 和 std_s_threadhold 自适应
        if len(self.mean_s_threadhold_q) >= 5:
            self.mean_s_threadhold = np.mean(self.mean_s_threadhold_q)
        if len(self.std_s_threadhold_q) >= 5:
            self.std_s_threadhold = np.mean(self.std_s_threadhold_q)

    def add_individual(self):
        if not (self.population.size < self.Ubound and self.population.size > self.Lbound):  # 当self.population_size 小于种群上界且大于种群上界
            return
        max_p_best_index=np.ceil(self.p*self.population.size)
        best_vectors=[]
        p_best_index=np.random.permutation(max_p_best_index)
        for i in range(self.change_size):
            p_best_vector = self.population.members[p_best_index[i]].vector
            for index,ele in p_best_vector: # 轻微扰动
                p_best_vector[index]+=np.random.uniform(-1.0/pow(1.0001,self.functionEvaluations),1.0/pow(1.0001,self.functionEvaluations))
            best_vectors.append(p_best_vector)
        self.population = np.append(self.population, np.array(best_vectors), axis=0)

    def substract_individual(self):
        if not (self.population.size < self.Ubound and self.population.size > self.Lbound):  # 当self.population_size 小于种群上界且大于种群上界
            return
        max_p_worst_index=np.ceil(self.p*self.population.size)
        p_worst_index=np.random.permutation(max_p_worst_index)
        p_worst_index=[x for x in p_worst_index if x!=0]
        for i in range(self.change_size):
            p_worst_vector=self.population.members[-p_worst_index[i]].vector
            for index,ele in enumerate(self.population):
                if (ele==p_worst_vector).any(axis=0):
                    self.population=np.delete(self.population,index,axis=0) # 删除平庸个体
                    # break #break掉，防止删除多个相同的population

    def substract_f(self):
        if self.f>0:
            self.f-=0.1

    def add_f(self):
        if self.f<2.0:
            self.f+=0.1

    def choose_action(self,state):
        self.check_state_exist(state)
        if np.random.uniform()<self.epsilon:
            state_action=self.q_table.loc[state,:]
            state_action=state_action.reindex(np.random.permutation(state_action.index)) #重置dataframe的索引（索引随机分配）
            action=state_action.idxmax()
        else:
            action=np.random.choice(self.actions)
        return action

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
        self.action=self.choose_action(str(self.state_prior))

        if self.action==0: # 增加种群
            self.add_entity()
            # self.add_entity2()
        elif self.action==1: # 减少种群
            self.substract_entity()
            # self.substract_entity2()
        elif self.action==2: # 更新变异率CR正态分布中值
            self.update_cr()
        elif self.action==3: # 增加F值正态分布中值
            self.add_f()
        elif self.action==4: # 减小F值正态分布中值
            self.substract_f()
        elif self.action==5: # 更新变异策略至最原始的DE变异策略
            self.update_to_raw_policy()
        elif self.action==6: # 更新变异策略至DE/best/1
            self.update_to_best1()
        elif self.action==7: # 更新变异策略至DE/rand/2
            self.update_to_rand2()
        elif self.action==8: # 更新变异策略至NSDE
            self.update_to_nsde()

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