# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:43:19 2022

@author: admin
"""

# load hmm_trials data to determine relationship between pupil diameter and hmm state.
# use cross validation to determine the appropriate dregree for a polynomial fitting relationship


import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import math

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]


base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

max_degree = 7
nKfold = 5
rsqs= np.full([len(hmm_trials_paths),max_degree,nKfold],np.nan)
step = .02
bins_lower = np.arange(0,1,step)
state_probs = np.full([len(hmm_trials_paths),3,len(bins_lower)],np.nan,dtype=np.float16)#[mouse,hmm_state,bin]        
for m, mouse_path in enumerate(hmm_trials_paths):
    mouse_path = hmm_trials_paths[m]
    hmm_trials = np.load(mouse_path,allow_pickle=True)
       
    # get all trials from sessions with optimal state present
    all_trials = pd.DataFrame()
    for i,trials in enumerate(hmm_trials):
        # if all([np.unique(trials['hmm_state'])[0]==0,np.isin('post_hoc_pupil_diameter',trials.columns)]):
        if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'post_hoc_pupil_diameter'
        
    this_measure = 'post_hoc_pupil_diameter'
    step = .02
    bins_lower = np.arange(0,1,step)
    bins_upper = np.arange(step,1+step,step)
        
    measure = np.array(all_trials[this_measure].values)
    state = np.array(all_trials['hmm_state'].values)
    
    # drop trials with indererminate state or nan values for pupil diameter
    measure = measure[state==state]
    state = state[state==state]
    state = state[measure==measure]
    measure = measure[measure==measure]
    
    
    kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)
    for ii, (train_index, test_index) in enumerate(kf.split(measure)):
        print(f"kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
            
        # split trials by kfolds
        train_measure = measure[train_index]
        train_state = state[train_index]
        test_measure = measure[test_index]
        test_state = state[test_index]
        
        
        #determine empirical relationship between pupil and optimal state occupancy for train and test sets
        train_opt_prob = np.full([len(bins_lower)],np.nan,dtype=np.float16)#[mouse,hmm_state,bin]        
        for i,this_bin in enumerate(bins_lower):
            lower = bins_lower[i]
            upper = bins_upper[i]      
            
            trial_ind = np.where(np.logical_and(train_measure>lower,train_measure<=upper))[0]          
            
            if len(trial_ind)>2:
                train_opt_prob[i] = len(np.where(train_state[trial_ind] == 0)[0])/len(trial_ind)   
        
        
        test_opt_prob = np.full([len(bins_lower)],np.nan,dtype=np.float16)#[mouse,hmm_state,bin]        
        for i,this_bin in enumerate(bins_lower):
            lower = bins_lower[i]
            upper = bins_upper[i]      
            
            trial_ind = np.where(np.logical_and(test_measure>lower,test_measure<=upper))[0]          
            
            if len(trial_ind)>3:
                test_opt_prob[i] = len(np.where(test_state[trial_ind] == 0)[0])/len(trial_ind)     


    # fit polynomials to training data and get test set r2 values 
        x= np.array(bins_lower,dtype=np.float32)
        y = np.array(train_opt_prob,dtype=np.float32)
        x = x[y==y]
        y = y[y==y]
        
        test_x = np.array(bins_lower,dtype=np.float32)
        test_y = np.array(test_opt_prob,dtype=np.float32)
        test_x = test_x[test_y==test_y]
        test_y = test_y[test_y==test_y]      
        for degree in np.arange(max_degree):
            coeffs1 = np.polyfit(x,y,degree+1)
            p1 = np.poly1d(coeffs1)
            rsqs[m,degree,ii] = r2_score(test_y, p1(test_x))
            # plt.subplot(nKfold,max_degree,degree+1+ii*max_degree)
            # plt.plot(test_x,test_y)
            # plt.plot(test_x,p1(test_x),'r')
            # plt.plot(x,y,'k')
            # plt.title(r2_score(test_y, p1(test_x)))
            rsqs[m,degree,ii] = r2_score(test_y, p1(test_x))
    
     
    for i,this_bin in enumerate(bins_lower):
        lower = bins_lower[i]
        upper = bins_upper[i]      
        
        trial_ind = np.where(np.logical_and(measure>lower,measure<=upper))[0]          
        
        if len(trial_ind)>5:
            state_probs[m,0,i] = len(np.where(state[trial_ind] == 0)[0])/len(trial_ind)   
            state_probs[m,1,i] = len(np.where(state[trial_ind] == 1)[0])/len(trial_ind)   
            state_probs[m,2,i] = len(np.where(state[trial_ind] > 1)[0])/len(trial_ind)   
        
        
    # if there are enough trials below lowest bin, combine them to include
    ind = np.where(state_probs[m,0,:]==state_probs[m,0,:])[0][0]-1
    trial_ind = np.where(measure<=bins_upper[ind])[0]        
    if len(trial_ind)>5:
        state_probs[m,0,ind] = len(np.where(state[trial_ind] == 0)[0])/len(trial_ind)   
        state_probs[m,1,ind] = len(np.where(state[trial_ind] == 1)[0])/len(trial_ind)   
        state_probs[m,2,ind] = len(np.where(state[trial_ind] > 1)[0])/len(trial_ind)   


# use elbow of r2 vs poly degree as degrees final fit 
poly_degree=[]
# plt.figure()
for m, mouse_rsqs in enumerate(rsqs):
    mouse_rsqs = rsqs[m,:,:]
    mouse_rsqs = np.mean(mouse_rsqs,axis=1)
    # plt.subplot(5,3,m+1)
    # plt.plot(mouse_rsqs,'k')
    # for i in range(1,7):
    i=6
    x1 = 0
    y1 = mouse_rsqs[0]
    x2 = i
    y2 = mouse_rsqs[i]
    b = (x1*y2 - x2*y1)/(x1-x2)
    m = (y1-y2)/(x1-x2)
    
    # plt.plot([0,i],[mouse_rsqs[0],m*x2 + b],'k')
        
    
    mins=[]
    signs=[]
    for ii in range(1,i):
        dist=[]
        dist2=[]
        p1 = np.array([ii,mouse_rsqs[ii]])
        for x in np.arange(0,i,.01):
            y = m*x+b
            p2 = np.array([x,y])
            dist.append(math.dist(p1,p2))
            dist2.append(np.linalg.norm(p1-p2))
        
        y=m*ii+b
        mins.append(min(dist))
        signs.append(mouse_rsqs[ii]-y)
        
        
        xx = np.argmin(dist)*.01
        yy = m*xx+b
        # plt.plot([ii,xx],[mouse_rsqs[ii],yy])

    this_degree = np.argmax(signs)+2
    poly_degree.append(this_degree)
    

np.save(base_folder + 'misc\\pupil_opt_polyfit_degrees.npy',poly_degree)
np.save(base_folder + 'misc\\pupil_opt_polyfit_rsqs.npy',rsqs)   
    

######## plot data 

# optimal_pupils=[]
# plt.figure()
# for m, mouse_rsqs in enumerate(rsqs):
#     mouse_rsqs = rsqs[m,:,:]
#     mouse_rsqs = np.mean(mouse_rsqs,axis=1)
#     if m <11:
#         sp_ind = m*2+1
#     elif m==11:
#         sp_ind = 25
#     elif m==12:
#         sp_ind = 29
            
    
#     plt.subplot(8,4,sp_ind+1)
#     mouse_state_probs = state_probs[m,:,:]
#     for i,state_prob in enumerate(mouse_state_probs):
#         plt.plot(bins_lower,state_prob,color=cols[i],zorder=10-i)
#         if i==0:
#             y= state_prob
#             x2=bins_lower[y==y]
#             y=y[y==y]    
#             opt_xs = np.arange(x2[0],x2[-1],.001)
#             this_poly_degree = poly_degree[m]
            
        
#             coeffs = np.polyfit(x2.astype(np.float32),y.astype(np.float32),this_poly_degree)
#             this_poly = np.poly1d(coeffs)
            
#             ys = this_poly(opt_xs)
#             this_opt_pupil = opt_xs[ys.argmax()].round(5)
#             optimal_pupils.append(this_opt_pupil)
            
#             plt.plot(opt_xs,ys,'k',zorder=10)
#             plt.plot([this_opt_pupil,this_opt_pupil],[0,1],'k--',zorder=10)
#             plt.xlim([0,1])
#             plt.ylabel(hmm_trials_paths[m][-20:-15])
        
#     plt.subplot(8,4,sp_ind)
#     plt.plot(np.arange(1,8),mouse_rsqs,'k')
#     # plt.ylim([0,1])
#     plt.plot(poly_degree[m],mouse_rsqs[poly_degree[m]-1],'sk')

# plt.subplot(3,4,11)

# plt.hist(poly_degree)
    

# plt.subplot(3,4,12)

# lineardata = rsqs[:,0,:].mean(axis=1)
# data_2deg = rsqs[:,1,:].mean(axis=1)
# all_rsqs = rsqs[:,:,:].mean(axis=2)
# data=[]
# for m,degree in enumerate(poly_degree):
#     data.append(all_rsqs[m,degree-1])
# plt.boxplot([lineardata,data_2deg,data],showfliers=False)
# x = np.ones(len(data))+np.random.randn(len(data))*.05
# plt.plot(x,lineardata,'ko')
# x = 1+np.ones(len(data))+np.random.randn(len(data))*.05
# plt.plot(x,data_2deg,'ko')
# x = 2+np.ones(len(data))+np.random.randn(len(data))*.05
# plt.plot(x,data,'ko')
    







