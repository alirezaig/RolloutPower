# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:25:56 2019

@author: alire
"""

#importing the packages
import numpy as np
import time
import pandas as pd
from Rollout_kstep_mod import *

# Setting and generating problem parameters
indx = 1
nfails = 15
ns = 1
nt = int(25) #Planning horizon

#initial state with 15 failures and 1 repair team
state_now = [1,0,0,0,2,0,0,0,0,3,0,0,0,0,0,6,0]

ts = int(nfails)  #All failures will happen during the first ts time periods
nb = 30         #Do not change this for experiments 
nl = 50         #Do not change this for experiments
stage_now = 0   #Do not change this for experiments
itr = 100   #Do not chage this for experiments
steps = 2

#Setting the random seed
ran_seed = 10
np.random.seed(ran_seed)
c = np.random.randint(300,500,nl+nb)
rt = np.random.randint(2,5,nl+nb)
rc = np.random.randint(50,100,nl+nb)
tt = np.random.randint(1,3,(nl+nb,nl+nb))
for i in range(nl+nb):
    for j in range(nl+nb):
        if i == j:
            tt[i,j] = 0
utc = np.random.uniform(1,3,ns)
#--------------------Defining Scenarios ------------------------------------------------------------------------------------------------
#scenariopr determines the number of hurricane scenarios and their liklihood
scenario_pr = [0.3,0.4,0.3]

#Here we define a list of lists corresponding to each scenario
#The first nested list gives the affected nodes, the second nested list, gives the hit time and the last nested list gives the probability
#of failure for that specific load if it is hit by the hurricane.
loads = list(range(0,nb))
tempintloads = list(np.random.permutation(len(loads))[0:nfails])
intloads = []
for j in tempintloads:
    intloads.append(loads[j])
nls = int(nfails/3)
sc1 = [intloads[0:nls],list(np.random.permutation(ts)[0:nls]+stage_now+1),list(np.random.uniform(0.4,0.8,nls))]
sc2 = [intloads[nls:2*nls],list(np.random.permutation(ts)[0:nls]+stage_now+1),list(np.random.uniform(0.4,0.8,nls))]
sc3 = [intloads[2*nls:len(intloads)],list(np.random.permutation(ts)[0:len(intloads)-2*nls]+stage_now+1),list(np.random.uniform(0.4,0.8,len(intloads)-2*nls))]

#Deriving the set of all nodes that could fail
#we first merge the lists of failed buses under each scenario
fbtemp = sc1[0] + sc2[0] + sc3[0]
#We then use hashtables to extract the uniqe elemets
hashtab = {}
for i in range(len(fbtemp)):
    if fbtemp[i] not in hashtab.values():
        hashtab.update({str(i):fbtemp[i]})
fb = list(hashtab.values())
nfb = len(fb)

#Generating a set of initial states
#num_init = 5
#init_states = np.zeros((num_init,nfb+2*ns))
#for i in range(num_init):
#    for j in range(nfb):
#        rand_num = np.random.uniform(0,1)
#        if rand_num <= 0.5:
#            init_states[i,j] = 0
#        else:
#            init_states[i,j] = int(np.random.uniform(0,1)*nt*0.5)
#    rand_perm = np.random.permutation(nfb)[0:ns]
#    for j in range(ns):
#        init_states[i,nfb+j*2] = rand_perm[j]
#        if init_states[i,int(rand_perm[j])] > 0:
#            init_states[i,nfb+j*2+1] = np.random.randint(
#                    max(1,rt[fb[int(rand_perm[j])]]-init_states[i,int(rand_perm[j])]),rt[fb[int(rand_perm[j])]])
#        else:
#            init_states[i,nfb+j*2] = 0 

ran_seed2 = [0,1,2,3,4,5,6,7,8,9]
results = pd.DataFrame([[-1]*4], columns = ['run','solution','tot_cost','run_time']) 
filename = '2step'+str(indx) 
filename2 = filename + '.csv'
for j in ran_seed2:
    tot_costs = []
    togo_means = []
    togo_vars = []
    run_times = []
    
    t1 = time.time()    
    a = rollout_kstep(state_now, stage_now,fb,tt,c,utc,rc,rt,nl,ns,nt,sc1,sc2,sc3,scenario_pr,itr,j,steps)
    t2 = time.time()
    run_times.append(t2-t1)
    run = [j]*len(a)
    sols = list(range(len(a)))
    run_time = [t2-t1]*len(a)
    df1 = {'run':run,'solution':sols,'tot_cost':a, 'run_time':run_time}            
    pd1  = pd.DataFrame(df1, columns=df1.keys())
    results = pd.concat([results, pd1], axis =0)
    results.to_pickle(filename)   
results.to_csv(filename2)
#results2 = pd.read_pickle(filename)
    
    
    
