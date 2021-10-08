# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:15:54 2021

@author: admin
"""

from uobrainflex.nwb import loadbehavior as load
from uobrainflex.behavioranalysis import flex_hmm
import matplotlib.pyplot as plt
import ssm
import numpy as np

#identify subject and find files in date range
subject='BW016'; sessions=[]
for fileID in range(20210421,20210428):
    this_session = load.get_file_path(subject,str(fileID))
    if this_session:
        sessions.append(this_session)
subject='BW041'; sessions=[]
for fileID in range(20210922,20210926):
    this_session = load.get_file_path(subject,str(fileID))
    if this_session:
        sessions.append(this_session)
subject='BW045'; sessions=[]
for fileID in range(20210915,20210926):
    this_session = load.get_file_path(subject,str(fileID))
    if this_session:
        sessions.append(this_session)

#load and format choice data for GLM-HMM
inpts=list([])
true_choices=list([])
sessions_trial_start =list([])
for sess in sessions:
    trial_data, trial_labels = load.load_trial_data(sess)
    these_inpts, these_true_choices, trial_start, _ = flex_hmm.format_choice_behavior_hmm(trial_data, trial_labels)
    inpts.extend(these_inpts)
    true_choices.extend([these_true_choices])
    sessions_trial_start.extend([trial_start])


### State Selection
#Create kfold cross-validation object which will split data for us
num_sess = len(true_choices)
from sklearn.model_selection import KFold
nKfold = np.min([num_sess,5]) # if there are more than 4 sessions, group sessions together to create 4 folds
synthetic_data=true_choices
synthetic_inpts=inpts

kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)

#Just for sanity's sake, let's check how it splits the data
#So 5-fold cross-validation uses 80% of the data to train the model, and holds 20% for testing
for ii, (train_index, test_index) in enumerate(kf.split(synthetic_data)):
    print(f"kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
max_states = 4 # largest number of states allowed in the model selection
N_iters = 1000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
TOL=10**-6 # tolerance parameter (see N_iters)

# initialized training and test loglik for model selection, and BIC
ll_training = np.zeros((max_states,nKfold))
ll_heldout = np.zeros((max_states,nKfold))
ll_training_map = np.zeros((max_states,nKfold))
ll_heldout_map = np.zeros((max_states,nKfold))
# BIC = np.zeros((max_states))

# Set the parameters of the GLM-HMM
num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

#Outer loop over the parameter for which you're doing model selection for
for iS, num_states in enumerate(range(1,max_states+1)):
    
    #Inner loop over kfolds
    for iK, (train_index, test_index) in enumerate(kf.split(synthetic_data)):
        nTrain = len(train_index); nTest = len(test_index)#*obs_dim
        
        # split choice data and inputs
        training_data = [synthetic_data[i] for i in train_index]
        test_data = [synthetic_data[i] for i in test_index]
        training_inpts=[synthetic_inpts[i] for i in train_index]
        test_inpts=[synthetic_inpts[i] for i in test_index]
        
    
        #Create HMM object to fit: MLE
        print(num_states)
        xval_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                           observation_kwargs=dict(C=num_categories), transitions="standard")
        #fit on training data
        hmm_lls = xval_glmhmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
        #Compute log-likelihood for each dataset
        ll_training[iS,iK] = xval_glmhmm.log_probability(training_data, inputs=training_inpts)/nTrain
        ll_heldout[iS,iK] = xval_glmhmm.log_probability(test_data, inputs=test_inpts)/nTest
        
        #Create HMM object to fit: MAP
        # Instantiate GLM-HMM and set prior hyperparameters
        prior_sigma = 2
        prior_alpha = 2
        map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                       observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                     transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
        #fit on training data
        hmm_lls = map_glmhmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
        #Compute log-likelihood for each dataset
        ll_training_map[iS,iK] = map_glmhmm.log_probability(training_data, inputs=training_inpts)/nTrain
        ll_heldout_map[iS,iK] = map_glmhmm.log_probability(test_data, inputs=test_inpts)/nTest
        
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#7E37B8']
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
for iS, num_states in enumerate(range(1,max_states+1)):
    plt.plot((iS+1)*np.ones(nKfold),ll_training[iS,:], color=cols[0], marker='o',lw=0)
    plt.plot((iS+1)*np.ones(nKfold),ll_heldout[iS,:], color=cols[1], marker='o',lw=0)
    plt.plot((iS+1)*np.ones(nKfold),ll_training_map[iS,:], color=cols[2], marker='o',lw=0)
    plt.plot((iS+1)*np.ones(nKfold),ll_heldout_map[iS,:], color=cols[3], marker='o',lw=0)

plt.plot(range(1,max_states+1),np.mean(ll_training,1), label="training_MLE", color=cols[0])
plt.plot(range(1,max_states+1),np.mean(ll_heldout,1), label="test_MLE", color=cols[1])
plt.plot(range(1,max_states+1),np.mean(ll_training_map,1), label="training_MAP", color=cols[2])
plt.plot(range(1,max_states+1),np.mean(ll_heldout_map,1), label="test_MAP", color=cols[3])
# plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
plt.legend(loc="lower right")
plt.xlabel("states")
plt.xlim(0, max_states+1)
plt.ylabel("Log Probability")
plt.show()
plt.title(subject + ' Model Fitting')