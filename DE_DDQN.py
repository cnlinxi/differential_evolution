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

np.random.seed(1)
tf.set_random_seed(1)

global INF
INF=float('inf') # 无穷大的数

DEBUG=True
DEBUG_FILE=True

sess=None
output_graph=True

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
    def __init__(self,n_features=10,*args, **kwargs): # n_features: 个体向量维数
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
        # self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64) # 初始 q_table
        self.mean_s_threadhold=5e7 # 自适应
        self.std_s_threadhold=1e9 # 自适用
        self.mean_s_threadhold_q=deque(maxlen=5)
        self.std_s_threadhold_q=deque(maxlen=5)

        self.actions_count=9 # 动作个数
        self.actions=list(range(self.actions_count))
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

        self.evolve_policy=0 # 保存当前的变异策略
        self.n_actions=7 # 动作个数
        self.actions=list(range(self.n_actions))
        self.action=np.random.choice(self.actions)
        self.n_features = n_features
        self.lr=0.01 # 学习率
        self.gamma=0.9 # 奖励衰减，使得算法能够看清大局
        self.epsilon_max=0.9 # 贪婪度
        self.replace_target_iter = 200
        self.memory_size = 3000 # 存储(s,[a,r],s_)的存储器大小
        self.batch_size = 32 # 更新q使用的采样样本数
        self.epsilon_increment = None
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.double_q=True # 是否启用DDQN
        self.current_trial_costs=[]

        self.min_max_scaler=MinMaxScaler()

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1',reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2',reuse=tf.AUTO_REUSE):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net',reuse=tf.AUTO_REUSE):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net',reuse=tf.AUTO_REUSE):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: self.min_max_scaler.fit_transform(observation)})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + \
                         0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: self.min_max_scaler.fit_transform(batch_memory[:, -self.n_features:]),    # next observation
                       self.s: self.min_max_scaler.fit_transform(batch_memory[:, -self.n_features:])})   # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.nn_cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: self.min_max_scaler.fit_transform(batch_memory[:, :self.n_features]),
                                                self.q_target: self.min_max_scaler.fit_transform(q_target)})
        self.cost_his.append(self.nn_cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

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
            self.lm = 0
        elif self.population.size <= self.Lbound:
            if self.lm<20:
                self.lm += 1
            self.um = 0
        if self.std_fitness - np.std(self.population.costs) > self.std_s_threadhold:
            if self.std_cd <20:
                self.std_cd += 1
            self.std_s = 0
        else:
            self.std_cd = 0
            if self.std_s <20:
                self.std_s += 1
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

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table=self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def update_population(self,action):
        if action==0: # 更新变异率CR正态分布中值
            self.update_cr()
        elif action==1: # 增加F值正态分布中值
            self.add_f()
        elif action==2: # 减小F值正态分布中值
            self.substract_f()
        elif action==3: # 更新变异策略至1
            self.update_to_policy1()
        elif action==4: # 更新变异策略至2
            self.update_to_policy2()
        elif action==5: # 更新变异策略至3
            self.update_to_policy3()
        elif action==6: # 更新变异策略至4
            self.update_to_policy4()

    def update_weight(self):
        self.mean_s_weight=1

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
        self.action=self.choose_action(member_i.vector)
        self.update_population(self.action)

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

    def generateTrialPopulation(self, np):
        n=len(self.strategies)
        for i in range(n):
            self.strategies[i]['crMemory'].append([])
        self.prior_scores=deepcopy(self.population.costs)
        trialMembers = []
        for i in range(np):
            trialMember = self.generateTrialMember(i)
            if self.absoluteBounds: # absoluteBounds设置为True，则范围在限定范围外的个体将被“纠正”
                trialMember.constrain(self.minVector, self.maxVector)
            trialMembers.append(trialMember)
            trial_member = self.assign_cost(trialMember)
            reward=-1
            if self.population.members[i].cost - trial_member.cost>0: #上次的cost比本次的cost大，正向奖励
                reward=1
            self.store_transition(self.population.members[i].vector,self.action,reward,trialMember.vector)
            self.learn() # learning
        return population.Population(members=trialMembers)

    def trialMemberSuccess(self, i, trialMember):
        self.strategies[trialMember.strategy]['crMemory'][-1].append(trialMember.cr)
        super(rlde,self).trialMemberSuccess(i,trialMember)

    def optimise(self):
        self.population = self.assignCosts(self.population)
        self.generation = 0
        while self.terminationCriterion() == False:
            self.generation += 1
            self.current_trial_members=[]
            trialPopulation = self.generateTrialPopulation(self.population.size) # 生成变异重组后的个体
            trialPopulation = self.assign_population_costs(trialPopulation) # 为变异重组后的种群评分，分数由population.members[i].cost = self.cost(member.vector)形式带回
            self.selectNextGeneration(trialPopulation) # 生成新种群
        return self.population.bestVector