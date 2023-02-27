# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:02:59 2022

@author: admin
"""
# load data & test set LL across states calculations to determine n_states and fit glm-hmm for each subject


import os
import numpy as np
import glob
from uobrainflex.behavioranalysis import flex_hmm
import ssm
import ray


def plateu(data,threshold=.001):
    ind = np.where(np.diff(data)>threshold)[0]
    ind[np.argmax(data[ind+1])]
    return ind[np.argmax(data[ind+1])]+1

@ray.remote
def MLE_hmm_fit(subject, num_states, training_inpts, training_choices):
    ## Fit GLM-HMM with MAP estimation:
    # Set the parameters of the GLM-HMM
    obs_dim = training_choices[0].shape[1]          # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(training_choices)))    # number of categories for output
    input_dim = inpts[0].shape[1]                                    # input dimensions

    TOL = 10**-4
    
    N_iters = 1000

    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                       observation_kwargs=dict(C=num_categories), transitions="standard")
    #fit on training data
    train_ll = hmm.fit(training_choices, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)   
    train_ll = train_ll[-1]/np.concatenate(training_inpts).shape[0]       
    return hmm, train_ll


base_folder = 'E:\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')
save_folder = base_folder

for m in range(len(hmm_trials_paths)):  
    subject = os.path.basename(hmm_trials_paths[m])[:5]   
   
    #load previously calculated log likelihoods across # states to determine number of states for this subject
    LL= np.load(glob.glob(base_folder+'n_states\\'+subject+'*MLE_test*')[-1])
    # average across folds, and take the best iteration of fitting for each # of states
    LL = LL.mean(axis=-1).max(axis=0) 
    # find plateu of state x LL function to determine n_states
    num_states = plateu(LL)+1
    
    # load hmm_trials variable and get trial inputs and choices
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle= True)
    true_choices = flex_hmm.get_true_choices_from_hmm_trials(hmm_trials)
    inpts = flex_hmm.get_inpts_from_hmm_trials(hmm_trials)
    
    mouse_hmms=[]
    mouse_LLs=[]
    processes=[]
    # fit 10 hmms and select best
    for z in range(10):
        processes.append(MLE_hmm_fit.remote(subject, int(num_states), inpts, true_choices))
        
    data = np.array([ray.get(p) for p in processes])
   
    
    hmm = data[np.argmax(data[:,1]),0]
    
    # input posterior probabilities to hmm_trials, and permute the hmm to match states across subjects
    probs,hmm_trials = flex_hmm.get_posterior_probs(hmm, hmm_trials)
    hmm = flex_hmm.permute_hmm(hmm, hmm_trials)
    probs,hmm_trials = flex_hmm.get_posterior_probs(hmm, hmm_trials, occ_thresh = .8)
    # flex_hmm.plot_state_psychometrics(subject,hmm_trials)
    # flex_hmm.p_optimal_by_feature(subject, hmm_trials)
    np.save(save_folder + 'hmm_trials\\' + subject + '_hmm_trials.npy',hmm_trials)
    np.save(save_folder + 'hmms\\' + subject + '_hmm.npy',hmm)
