# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:36:14 2022

@author: admin
"""

####### use crossvalidation to determine log likelihood of test-set fits of models across #of states


import os
import numpy as np
import glob
from uobrainflex.behavioranalysis import flex_hmm
from sklearn.model_selection import KFold
import time
import ssm
import ray



@ray.remote
def MAP_hmm_fit(subject, num_states, training_inpts, training_choices, test_inpts, test_choices):
    ## Fit GLM-HMM with MAP estimation:
    # Set the parameters of the GLM-HMM
    obs_dim = training_choices[0].shape[1]          # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(training_choices)))    # number of categories for output
    input_dim = inpts[0].shape[1]                                    # input dimensions

    TOL = 10**-4
    N_iters = 1000

    # Instantiate GLM-HMM and set prior hyperparameters (MAP version)
    prior_sigma = 2
    prior_alpha = 2
    
    #generate hmm object and fit to data
    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                         observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                         transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    
    train_ll = hmm.fit(training_choices, inputs=training_inpts, method="em", num_iters=N_iters,
                                tolerance=TOL)   
    train_ll = train_ll[-1]/np.concatenate(training_inpts).shape[0]
    test_ll = hmm.log_probability(test_choices,test_inpts)/np.concatenate(test_inpts).shape[0]
    return hmm, train_ll, test_ll

@ray.remote
def MLE_hmm_fit(subject, num_states, training_inpts, training_choices, test_inpts, test_choices):
    ## Fit GLM-HMM with MLE estimation:
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
    test_ll = hmm.log_probability(test_choices,test_inpts)/np.concatenate(test_inpts).shape[0]
    return hmm, train_ll, test_ll


base_folder = 'E:\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\' + '*hmm_trials.npy')

max_states = 7
nKfold = 5
initializations = 10

for m in range(1,len(hmm_trials_paths)): # for each subject
    #build blank variabiles to fill
    MAP_train_LL= np.full([initializations,max_states,nKfold],np.nan)
    MAP_test_LL= np.full([initializations,max_states,nKfold],np.nan)
    MAP_HMM= np.full([initializations,max_states,nKfold],ssm.HMM(1,1))
    MAP_processes=[]
    
    MLE_train_LL= np.full([initializations,max_states,nKfold],np.nan)
    MLE_test_LL= np.full([initializations,max_states,nKfold],np.nan)
    MLE_HMM= np.full([initializations,max_states,nKfold],ssm.HMM(1,1))
    
    MLE_processes=[]
    
    fold=[]
    states=[]
    init=[]
    
    #load previously created hmm_trials variable
    mouse_path = hmm_trials_paths[m]
    print('mouse ' + str(m+1) + ' of ' + str(len(hmm_trials_paths)))
    subject = os.path.basename(mouse_path)[:5]
    hmm_trials = np.load(mouse_path,allow_pickle=True) 
    
    #get inputs and true choices from hmm_trials varaible
    inpts = flex_hmm.get_inpts_from_hmm_trials(hmm_trials)
    true_choices = flex_hmm.get_true_choices_from_hmm_trials(hmm_trials)
    
    kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)
    #Just for sanity's sake, let's check how it splits the data
    for ii, (train_index, test_index) in enumerate(kf.split(true_choices)):
        print(f"kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
    
    #loop over kfolds
    for iK, (train_index, test_index) in enumerate(kf.split(true_choices)):
        nTrain = len(train_index); nTest = len(test_index)#*obs_dim
            
        # split choice data and inputs to training and test sets
        training_choices = [true_choices[i] for i in train_index]
        test_choices = [true_choices[i] for i in test_index]
        training_inpts=[inpts[i] for i in train_index]
        test_inpts=[inpts[i] for i in test_index]

        #use ray to parallel process iterations
        for num_states in range(1,max_states+1):
            for z in range(initializations):
                MLE_processes.append(MLE_hmm_fit.remote(subject, num_states, training_inpts, training_choices, test_inpts, test_choices))
                MAP_processes.append(MAP_hmm_fit.remote(subject, num_states, training_inpts, training_choices, test_inpts, test_choices))
                fold.append(iK)
                states.append(num_states)
                init.append(z)
        
    # get kfold and initializaiton data
    MLE_results = np.array([ray.get(p) for p in MLE_processes])
    MAP_results = np.array([ray.get(p) for p in MAP_processes])
    
    # split results
    for i in range(len(init)):
        MAP_HMM[init[i],states[i]-1,fold[i]] = MAP_results[i,0]
        MAP_train_LL[init[i],states[i]-1,fold[i]] = MAP_results[i,1]
        MAP_test_LL[init[i],states[i]-1,fold[i]] = MAP_results[i,2]
        
        MLE_HMM[init[i],states[i]-1,fold[i]] = MLE_results[i,0]
        MLE_train_LL[init[i],states[i]-1,fold[i]] = MLE_results[i,1]
        MLE_test_LL[init[i],states[i]-1,fold[i]] = MLE_results[i,2]
        
    

    
    file_id = round(time.time())
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MLE_hmms_' + str(file_id),MLE_HMM)
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MLE_test_LL_' + str(file_id),MLE_test_LL)
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MLE_train_LL_' + str(file_id),MLE_train_LL)
    
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MAP_hmms_' + str(file_id),MAP_HMM)
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MAP_test_LL_' + str(file_id),MAP_test_LL)
    np.save(base_folder + 'n_states\\' + subject + '_state_testing_MAP_train_LL_' + str(file_id),MAP_train_LL)


