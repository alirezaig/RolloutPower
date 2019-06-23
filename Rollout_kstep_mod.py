# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:25:37 2019

@author: alire
"""

import numpy as np
import time
import pandas as pd
import math
from scipy.stats import norm
from gurobipy import *
from itertools import *
from Rolloutmod import *
#This function is used to maniplulate the lists of lists in other functions
def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList 
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList

#This function takes an assignment as input and generates the action matrix as output 
def ass_to_mat(state,assign,fb,ns):
    nfb = len(fb)
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
            
            
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_index.append(i)
            fail_node.append(fb[i])
            
    N = len(fail_node)
    M = len(avail_res)
    k = min(M,N)
    if k == 0:
        return np.matrix(np.ones((2,2))*-1)
    else:
        action_matrix = np.matrix(np.zeros((N,M)))
        if N >= M:
            for j in range(len(assign)):
                action_matrix[int(assign[j]),j] = 1
        else:
            for j in range(len(assign)):
                action_matrix[j,int(assign[j])] = 1
        return action_matrix

#Given the current system state this function runs the heuristic ploicy to select an action
def hueris(state,fb,c,tt,rt,utc,nl,ns,rc):
    nfb = len(fb)
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
            
            
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_index.append(i)
            fail_node.append(fb[i])
            
    N = len(fail_node)
    M = len(avail_res)
    k = min(M,N)
    if k == 0:
        return np.matrix(np.ones((2,2))*-1)
    
    #If we have atleast one free resoruce and one failed load, we decide on the assignment by formulating the model
    #We form the dictionaries to extract and store problem parameters
    c_mo = {}
    tf_mo = {}
    tt_mo = {}
    tt_no = {}
    rt_mo = {}
    rc_mo = {}
    ta_temp = {}
    tc_mo = {}
    for i in range(N):
        c_mo[i] = c[int(nl+fail_node[i])]
        rt_mo[i] = rt[int(nl+fail_node[i])]
        rc_mo[i] = rc[int(nl+fail_node[i])]
        tf_mo[i] = state[int(fail_index[i])]
        for j in range(M):
            tt_mo[i,j] = tt[int(nl+fail_node[i]),int(nl+res_loc[j])]
        for k in range(N):
            tt_no[i,k] = tt[int(nl+fail_node[i]),int(nl+fail_node[k])]
    for j in range(M):
        tc_mo[j] = utc[int(avail_res[j])]
    #Creating the resource assignment model
    mod1 = Model()
     
    #Defining the variables
    x = mod1.addVars(((i,j) for i in range(N) for j in range(M)), vtype = GRB.BINARY)
    ta_temp = mod1.addVars(((i,j) for i in range(N) for j in range(M)), vtype = GRB.CONTINUOUS)
    ta = mod1.addVars([i for i in range(N)], vtype = GRB.CONTINUOUS)

    #Adding the constraints
    for j in range(M):
        mod1.addConstr(quicksum(x[i,j] for i in range(N)) <= 1)
    for i in range(N):
        mod1.addConstr(quicksum(x[i,j] for j in range(M)) <= 1)
    mod1.addConstr(quicksum(x[i,j] for i in range(N) for j in range(M)) == min(M,N))
    
    #The following constraint first derives the soonest time until load i can be fixed by resource j
    for i in range(N):
        for j in range(M):
            mod1.addConstr(ta_temp[i,j] == quicksum(x[k,j]*(tt_mo[k,j]+rt_mo[k]+tt_no[k,i]+rt_mo[i]) for k in range(N)))
    #The following constraints dictate that the time to fix take one of the values and objective function makes it to pick the smallest value
    mod1.addConstrs(ta[i] <= ta_temp[i,j] for j in range(M) for i in range(N))

    #Defining the objective function
    #mod1.setObjective(quicksum(x[i,j]*(c_mo[i]*(1+rt_mo[i]) - utc*tt_mo[i,j] - rc_mo[i]) for i in range(N) for j in range(M)), GRB.MAXIMIZE)
    mod1.setObjective(quicksum(x[i,j]*(c_mo[i]*(tf_mo[i]+rt_mo[i])-rc_mo[i]-tt_mo[i,j]*tc_mo[j])for i in range(N) for j in range(M)) 
    + quicksum(c[i]*ta[i]*(1-quicksum(x[i,j] for j in range(M))) for i in range(N)), GRB.MAXIMIZE)
    mod1.update
    mod1.optimize()
    assi_temp = np.matrix(np.reshape((mod1.getAttr ('x',x).values()),(N,M)))
    return assi_temp

#This function takes the state, action and scenario and comes up with the next state
def next_state(state,stage,sc,fb,nl,ns,rt,tt,action_mat):  
    nfb = len(fb)
    new_state = np.copy(state)
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_index.append(i)
            fail_node.append(fb[i])
    
    #Here we first apply the effect of the action that we have taken
    if int(np.sum(action_mat)) != 0:
        for i in range(action_mat.shape[1]):
            res_col = np.copy(action_mat[:,i])
            if np.sum(res_col) == 1:
                node_temp = fail_node[int(np.argmax(res_col))]
                node = fb.index(node_temp)
                res = avail_res[i]
                new_state[nfb+(2*res)+1] = rt[nl+int(fb[node])] + tt[nl+int(fb[node]),fb[ int(new_state[nfb+(2*res)])]]
                new_state[nfb+(2*res)] = node
    #Here we apply the updates just due to time that passes
    for i in range(nfb):
        if new_state[i] > 0:
            new_state[i] = new_state[i] + 1
    #Here we apply the updates due to restoration process
    for j in range(ns):
        if new_state[nfb+(2*j)+1] == 1:
            loc = new_state[nfb+(2*j)]
            new_state[int(loc)] = 0
        if new_state[nfb+(2*j)+1] > 0:
            new_state[nfb+(2*j)+1] = new_state[nfb+(2*j)+1] - 1  
    #Here we apply the updates due to failures that might happen according to the current scenario
    if stage in sc[1]:
        nodes = list.copy(sc[0])
        times = list.copy(sc[1])
        probs = list.copy(sc[2])
        if stage in times:
            node = nodes[int(times.index(stage))]
            node_index = fb.index(node)
            prob = probs[int(times.index(stage))]
            randnum = np.random.uniform(0,1)
            if randnum <= prob and new_state[node_index] == 0:
                new_state[node_index] = 1
    return new_state

#Here we define a function that takes the state and caclulates the incurred cost for in one stage
def inccost(state,action_mat,tt,c,fb,utc,rc,ns,nl):    
    z = 0
    nfb = len(fb)
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_index.append(i)
            fail_node.append(fb[i])
    if int(np.sum(action_mat)) != -4:
        for k2 in range(action_mat.shape[1]):
            res_col = np.copy(action_mat[:,k2])
            res_index = avail_res[k2]
            if np.sum(res_col) == 1:
                node = fail_index[int(np.argmax(res_col))]
                z = z + tt[nl+int(fb[node]),fb[ int(state[nfb+(2*res_index)])]]*utc[res_index] + rc[nl+int(fb[node])]     
    for i in range(nfb):
        if state[i] > 0:
            z = z + c[nl+int(fb[i])]
    return z
   
#Here we define a function that caculates approximated cost-to-go
def costtogo_hat(state,stage,fb,c,tt,rt,utc,nl,ns,nt,sc,itr,ran_seed,rc):    
    np.random.seed(ran_seed)
    costs = np.zeros(itr)
    for i in range(itr):
        z = 0
        states_path = {}
        states_path[stage] = np.copy(state)
        for j in range(stage+1,nt-1):
            current_state = np.copy(states_path[j-1])
            opt_action_mat = hueris(current_state,fb,c,tt,rt,utc,nl,ns,rc)
            z = z + inccost(current_state,opt_action_mat,tt,c,fb,utc,rc,ns,nl)
            states_path[j] = next_state(current_state,j-1,sc,fb,nl,ns,rt,tt,opt_action_mat)
        costs[i] = z
    return np.average(costs),np.var(costs) 

#Given the current system state this function generates the set of all possible actions
def genactions(state,fb,ns):
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    nfb = len(fb)
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_node.append(fb[i])
            fail_index.append(i)
    N = len(fail_node)
    M = len(avail_res)
    k = min(N,M)
    l = max(N,M)
    if k > 0:
        subsets = list(itertools.combinations(set(range(l)),k))
        assigns_temp = []
        for i in range(len(subsets)):
            set1 = list(subsets[i])
            set2 = list(range(k))
            new_assi = list(itertools.permutations(set1))
            assigns_temp.append(new_assi)
            assigns_temp = flatten(assigns_temp)
        assigns = []
        for i in range(len(assigns_temp)):
            assigns.append(list(assigns_temp[i]))
        return assigns
    else:
        return np.matrix(np.ones((2,2))*-1)

#Here we define a function that takes the current stage, current state and scenarios and generates all possible next states
def possible_states(state,stage,nl,sc1,sc2,sc3,fb,ns,rt,tt): 
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    nfb = len(fb)
    for j in range(ns):
        if state[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state[nfb+(2*j)])
        else:
            rep_index.append(state[nfb+(2*j)])
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_node.append(fb[i])
            fail_index.append(i)
    N = len(fail_node)
    M = len(avail_res)
    k = min(M,N)
    next_states = set()
    if k>0:
        possible_actions = genactions(state,fb,ns)
        for i in range(len(possible_actions)):
            assign = possible_actions[i]
            action_mat = ass_to_mat(state,assign,fb,ns)
            new_state = np.copy(state)
            #Here we apply the effect of the action that we have taken
            for j in range(action_mat.shape[1]):
                res_col = np.copy(action_mat[:,j])
                res_index = avail_res[j]
                if np.sum(res_col) == 1:
                    node = fail_index[int(np.argmax(res_col))]
                    new_state[nfb+(2*res_index)+1] = rt[nl+int(fb[node])] + tt[nl+int(fb[node]),fb[ int(new_state[nfb+(2*res_index)])]]
                    new_state[nfb+(2*res_index)] = node
            #Here we apply the updates just due to time that passes
            for j in range(nfb):
                if new_state[j] > 0:
                    new_state[j] = new_state[j] + 1
            #Here we apply the updates due to restoration process
            for j in range(ns):
                if new_state[nfb+(2*j)+1] == 1:
                    loc = int(new_state[nfb+(2*j)])
                    new_state[loc] = 0
                if new_state[nfb+(2*j)+1] > 0:
                    new_state[nfb+(2*j)+1] = new_state[nfb+(2*j)+1] - 1 
            next_states.add(tuple(new_state))
            #For each scenario we need to have the following code to have the complete list of new states
            if stage in sc1[1]:
                new_state2 = np.copy(new_state)
                nodes = list.copy(sc1[0])
                times = list.copy(sc1[1])
                node = fb.index(nodes[int(times.index(stage))])
                if new_state2[node] == 0:
                    new_state2[node] = 1
                    next_states.add(tuple(new_state2))
                                
            if stage in sc2[1]:
                new_state2 = np.copy(new_state)
                nodes = list.copy(sc2[0])
                times = list.copy(sc2[1])
                node = fb.index(nodes[int(times.index(stage))])
                if new_state2[node] == 0:
                    new_state2[node] = 1
                    next_states.add(tuple(new_state2))
                            
            if stage in sc3[1]:
                new_state2 = np.copy(new_state)
                nodes = list.copy(sc3[0])
                times = list.copy(sc3[1])
                node = fb.index(nodes[int(times.index(stage))])
                if new_state2[node] == 0:
                    new_state2[node] = 1
                    next_states.add(tuple(new_state2))
    else:
        new_state = np.copy(state)
        #Here we apply the effect just to the time that passes
        for j in range(nfb):
            if new_state[j] > 0:
                        new_state[j] = new_state[j] + 1
        #Here we apply the updates due to restoration process
        for j in range(ns):
            if new_state[nfb+(2*j)+1] == 1:
                loc = int(new_state[nfb+(2*j)])
                new_state[loc] = 0
            if new_state[nfb+(2*j)+1] > 0:
                new_state[nfb+(2*j)+1] = new_state[nfb+(2*j)+1] - 1 
        next_states.add(tuple(new_state))
        #For each scenario we need to have the following code to have the complete list of new states
        if stage in sc1[1]:
            new_state2 = np.copy(new_state)
            nodes = list.copy(sc1[0])
            times = list.copy(sc1[1])
            node = fb.index(nodes[int(times.index(stage))])
            if new_state2[node] == 0:
                    new_state2[node] = 1
                    next_states.add(tuple(new_state2))
                    
        if stage in sc2[1]:
            new_state2 = np.copy(new_state)
            nodes = list.copy(sc2[0])
            times = list.copy(sc2[1])
            node = fb.index(nodes[int(times.index(stage))])
            if new_state2[node] == 0:
                new_state2[node] = 1
                next_states.add(tuple(new_state2))
                
        if stage in sc3[1]:
            new_state2 = np.copy(new_state)
            nodes = list.copy(sc3[0])
            times = list.copy(sc3[1])
            node = fb.index(nodes[int(times.index(stage))])
            if new_state2[node] == 0:
                new_state2[node] = 1
                next_states.add(tuple(new_state2))
    return next_states
def costtogo(state,stage,nl,sc1,sc2,sc3,fb,action,ns,sc_pr,states_values,rt,tt,c,utc,rc):   
    z = 0
    next_states = []
    new_state = np.copy(state)
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    nfb = len(fb)
    next_states = []
    for j in range(ns):
            if state[nfb+1+(2*j)] == 0:
                avail_res.append(j)
                res_loc.append(state[nfb+(2*j)])
            else:
                rep_index.append(state[nfb+(2*j)])
    for i in range(nfb):
        if state[i] > 0 and i not in rep_index:
            fail_node.append(fb[i])
            fail_index.append(i)
    #Here we apply the effect of the action that we have taken        
    if int(np.sum(action)) != -4:
        for j in range(action.shape[1]):
            res_col = np.copy(action[:,j])
            res_index = avail_res[j]
            if np.sum(res_col) == 1:
                node = fail_index[int(np.argmax(res_col))]
                new_state[nfb+(2*res_index)+1] = rt[nl+int(fb[node])] + tt[nl+int(fb[node]),fb[ int(new_state[nfb+(2*res_index)])]]
                new_state[nfb+(2*res_index)] = node        
    #Here we apply the updates just due to time that passes
    for j in range(nfb):
        if new_state[j] > 0:
                new_state[j] = new_state[j] + 1
    #Here we apply the updates due to restoration process
    for j in range(ns):
        if new_state[nfb+(2*j)+1] == 1:
            loc = int(new_state[nfb+(2*j)])
            new_state[loc] = 0
        if new_state[nfb+(2*j)+1] > 0:
            new_state[nfb+(2*j)+1] = new_state[nfb+(2*j)+1] - 1    
    #Here we obtain the possible new states and thier associated probabilities and store them in lists
    next_states = []
    prs = []
    prs.append(0)
    next_states.append(new_state)
    #For each scenario we need to have the following code to have the complete list of new states
    if stage in sc1[1]:
        new_state2 = np.copy(new_state)
        nodes = list.copy(sc1[0])
        times = list.copy(sc1[1])
        probs = list.copy(sc1[2])
        node = fb.index(nodes[int(times.index(stage))])
        prob = probs[int(times.index(stage))]
        if new_state2[node] == 0:
            new_state2[node] = 1
            next_states.append(new_state2)
            prs.append(prob*sc_pr[0])
            prs[0] += sc_pr[0]*(1-prob)
        else:
            prs[0] += sc_pr[0]
    else:
        prs[0] += sc_pr[0]
    if stage in sc2[1]:
        new_state2 = np.copy(new_state)
        nodes = list.copy(sc2[0])
        times = list.copy(sc2[1])
        probs = list.copy(sc2[2])
        node = fb.index(nodes[int(times.index(stage))])
        prob = probs[int(times.index(stage))]
        if new_state2[node] == 0:
            new_state2[node] = 1
            next_states.append(new_state2)
            prs.append(prob*sc_pr[1])
            prs[0] += sc_pr[1]*(1-prob)
        else:
            prs[0] += sc_pr[1] 
   
    else:
        prs[0] += sc_pr[1]
        
    if stage in sc3[1]:
        new_state2 = np.copy(new_state)
        nodes = list.copy(sc3[0])
        times = list.copy(sc3[1])
        probs = list.copy(sc3[2])
        node = fb.index(nodes[int(times.index(stage))])
        prob = probs[int(times.index(stage))]
        if new_state2[node] == 0:
            new_state2[node] = 1
            next_states.append(new_state2)
            prs.append(prob*sc_pr[2])
            prs[0] += sc_pr[2]*(1-prob)
        else:
            prs[0] += sc_pr[2] 
    else:
        prs[0] += sc_pr[2]
    #Here having all possible states in next stage and their associated probabbilities, we calculate the cost to go
    for i in range(len(prs)):
        if type(prs[i]) is list:
            pro = prs[i][0]
        else:
            pro = prs[i]
        if pro > 0:
            z = z + pro*states_values[(stage+1,tuple(next_states[i]))]
    z = z + inccost(state,action,tt,c,fb,utc,rc,ns,nl)
    return z       


#Using the functions coded above this function uses the Rollout to solve a problem starting from 
def rollout(state_now, stage_now,fb,tt,c,utc,rc,rt,nl,ns,nt,sc1,sc2,sc3,scenario_pr,iters,ran_seed):
    #We first form the list of available resources and failed node    
    avail_res = []
    res_loc = []
    fail_node = []
    fail_index = []
    rep_index = []
    nfb = len(fb)
    for j in range(ns):
        if state_now[nfb+1+(2*j)] == 0:
            avail_res.append(j)
            res_loc.append(state_now[nfb+(2*j)])
        else:
            rep_index.append(state_now[nfb+(2*j)]) 
    for i in range(nfb):
        if state_now[i] > 0 and i not in rep_index:
            fail_node.append(fb[i])
            fail_index.append(i)
    
    N = len(fail_node)
    M = len(avail_res)
    k = min(N,M)
    l = max(N,M)
    vals = []
    assignments = []
    togo_avg = []
    togo_var = []
    if k>0:   
        subsets = list(itertools.combinations(set(range(l)),k))
        assigns = []
        for i in range(len(subsets)):
            set1 = list(subsets[i])
            set2 = list(range(k))
            new_assi = list(itertools.permutations(set1))
            assigns.append(new_assi)
        for i in range(len(assigns)):
            assign = np.copy(assigns[i])
            for j in range(len(assign)):
                assign_final = np.copy(assign[j])
                action_mat_final = ass_to_mat(state_now,assign_final,fb,ns)
                
                #The action we take incurs some costs to the system
                cost_now = inccost(state_now,action_mat_final,tt,c,fb,utc,rc,ns,nl)
                #The action we take also takes the system to a new state
                state_fut = np.copy(state_now)
                for k2 in range(action_mat_final.shape[1]):
                    res_col = np.copy(action_mat_final[:,k2])
                    if np.sum(res_col) == 1:
                        node_temp = fail_node[int(np.argmax(res_col))]
                        node = fb.index(node_temp)
                        res = avail_res[k2]
                        state_fut[nfb+(2*res)+1] = rt[nl+int(fb[node])] + tt[nl+int(fb[node]),fb[ int(state_fut[nfb+(2*res)])]]
                        state_fut[nfb+(2*res)] = node
                out_sc1 = costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc1,iters,ran_seed,rc)
                out_sc2 = costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc2,iters,ran_seed,rc)
                out_sc3 =  costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc3,iters,ran_seed,rc)                
                cost = cost_now + scenario_pr[0]*out_sc1[0] + scenario_pr[1]*out_sc2[0] + scenario_pr[2]*out_sc3[0]
                togo_avg.append(scenario_pr[0]*out_sc1[0] + scenario_pr[1]*out_sc2[0] + scenario_pr[2]*out_sc3[0])
                togo_var.append((scenario_pr[0]**2)*out_sc1[1] + (scenario_pr[1]**2)*out_sc2[1] + (scenario_pr[2]**2)*out_sc3[1])
                vals.append(cost)
                assignments.append(assign_final)
    else:
        state_fut = np.copy(state_now)
        out_sc1 = costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc1,iters,ran_seed,rc)
        out_sc2 = costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc2,iters,ran_seed,rc)
        out_sc3 =  costtogo_hat(state_fut,stage_now+1,fb,c,tt,rt,utc,nl,ns,nt,sc3,iters,ran_seed,rc)
        cost_now = inccost(state_now,np.matrix(np.ones((2,2))*-1),tt,c,fb,utc,rc,ns,nl)
        cost = cost_now+scenario_pr[0]*out_sc1[0] + scenario_pr[1]*out_sc2[0] + scenario_pr[2]*out_sc3[0]
        togo_avg.append(scenario_pr[0]*out_sc1[0] + scenario_pr[1]*out_sc2[0] + scenario_pr[2]*out_sc3[0])
        togo_var.append((scenario_pr[0]**2)*out_sc1[1] + (scenario_pr[1]**2)*out_sc2[1] + (scenario_pr[2]**2)*out_sc3[1])
        vals.append(cost)
    return vals      

#Using the functions coded above this function uses the Rollout to solve a problem starting from 
def rollout_kstep(state_now, stage_now,fb,tt,c,utc,rc,rt,nl,ns,nt,sc1,sc2,sc3,scenario_pr,iters,ran_seed,k):    
    #Here we form the intitial state of the network and using the initial state we form a panda dataframe that is used 
    #later on to store possible states        
    initstate = np.copy(state_now)
    initstate = pd.Series(initstate)
    stages_states = pd.DataFrame(initstate)
    stages_states = stages_states.transpose()
    stages_states['Stage'] = [stage_now]
    #Here starting from the first stage, we extract the current states from the panda dataframe then using 
    #the function defined above we obtain possible states in the next stage and append them to the dataframe.
    if k > 1:
        for i in range(stage_now,stage_now+k-1):
            temp = stages_states.loc[stages_states['Stage'] == i]
            current_states = temp.drop(temp.columns[-1], axis=1)
            nstates = len(current_states.index)
            next_states = set()
            for j in range(nstates):
                state = list(current_states.iloc[j])
                new_states = possible_states(state,i,nl,sc1,sc2,sc3,fb,ns,rt,tt)
                next_states = next_states | new_states
            next_states2 = pd.DataFrame(list(next_states))
            next_states2['Stage'] = [i+1]*len(next_states2.index)
            stages_states = stages_states.append(next_states2)
        states_values = {}
    
    #Calcuting the incurred cost at last stage, (solving the trivial problem)
    #We fist extract the possible states in the last stage from the the dataframe
    temp = stages_states.loc[stages_states['Stage'] == stage_now+k-1]
    current_states = temp.drop(temp.columns[-1], axis=1)
    nstates = len(current_states.index)
    for j in range(nstates):
        final_state = list.copy(list(current_states.iloc[j]))
        costs = rollout(final_state, stage_now+k-1,fb,tt,c,utc,rc,rt,nl,ns,nt,sc1,sc2,sc3,scenario_pr,iters,ran_seed)
        if k > 1:
           states_values[(stage_now+k-1,tuple(final_state))] = min(costs)
        else:
            return costs
        
    #calculating the value function for previous stages
    for i in range(stage_now+k-2,stage_now-1,-1):
        temp = stages_states.loc[stages_states['Stage'] == i]
        current_states = temp.drop(temp.columns[-1], axis=1)
        nstates = len(current_states.index)
        for j in range(nstates):
            new_state = np.copy(np.array((current_states.iloc[j])))
            possible_actions = genactions(new_state,fb,ns)
            values = []
            if int(np.sum(possible_actions)) != -4:
                for k2 in range(len(possible_actions)):
                    new_action = ass_to_mat(new_state,possible_actions[k2],fb,ns)
                    values.append(costtogo(
                            new_state,i,nl,sc1,sc2,sc3,fb,new_action,ns,scenario_pr,states_values,rt,tt,c,utc,rc))
                states_values[(i,tuple(new_state))] = min(values)
            else:
                cost = costtogo(
                        new_state,i,nl,sc1,sc2,sc3,fb,np.matrix(np.ones((2,2))*-1),ns,scenario_pr,states_values,rt,tt,c,utc,rc)
                costtogo(state,stage,nl,sc1,sc2,sc3,fb,action,ns,sc_pr,states_values,rt,tt,c,utc,rc)
                states_values[(i,tuple(new_state))] = cost
                values.append(cost)
    return values    