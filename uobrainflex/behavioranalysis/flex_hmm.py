# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:49:15 2021 by Daniel Hulsey

Functions for formatting task performance data for use with SSM package 
"""

import numpy as np

def format_choice_behavior_hmm(trial_data, trial_labels, drop_no_response=True):
    """
    Parameters
    ----------
    trial_data : dataframe
        output of load_trials for a single session.
    drop_no_response : Boolean
        indicates whether to drop no response trials
    Returns
    -------
    inpts : TYPE
        DESCRIPTION.
    true_choice : TYPE
        DESCRIPTION.

    """
    miss_ind = trial_labels['outcome']['incorrect_reject']
    right_stim = trial_labels['target_port']['right']
    left_stim = trial_labels['target_port']['left']
    hit = trial_labels['outcome']['hit']
    error = trial_labels['outcome']['miss']
    
    if drop_no_response:
        # remove no response trials
        response_trials = trial_data.drop(index = trial_data.index[trial_data['outcome']==miss_ind].tolist())
        response_inds = response_trials.index #get indices of kept trial data
    else:
        response_inds = trial_data.index#get ALL indices of trial data
    trial_start = response_trials['start_time'].values
    # create target port ind where left = -1 and right = 1
    target_port = response_trials['target_port']
    target_port = target_port.replace(left_stim,-1)
    target_port = target_port.replace(right_stim,1)
    target_port = target_port.values
    
    # get coherences of toneclouds presented
    TC_coherence = response_trials['auditory_stim_difficulty'].values
    
    # multiply coherence by stim id to get inpts variable. this should be shifted to match coherence calculation in the future
    stim = target_port * TC_coherence
    
    # create true_choice array. This should be created during labview data collection in the future
    true_choice = np.full([len(response_trials),1],np.int(0)) 
    # step through each response and update true_choice
    for ind in range(len(response_trials)):
        # if hit and left stim
        if all([response_trials['outcome'].iloc[ind] == hit, response_trials['target_port'].iloc[ind] == left_stim]):
            true_choice[ind] = int(0)
        # if error and right stim
        if all([response_trials['outcome'].iloc[ind] == error, response_trials['target_port'].iloc[ind] == right_stim]):
            true_choice[ind] = int(0)
        # if hit and right stim
        if all([response_trials['outcome'].iloc[ind] == hit, response_trials['target_port'].iloc[ind] == right_stim]):    
            true_choice[ind] = int(1)
        # if error and left stim
        if all([response_trials['outcome'].iloc[ind] == error, response_trials['target_port'].iloc[ind] == left_stim]):    
            true_choice[ind] = int(1)
        if response_trials['outcome'].iloc[ind] == miss_ind:#if the outcome is a miss, true choice = 2    
            true_choice[ind] = int(2)
    
    # create list of choices with [stimulus information, bias constant]
    inpts = [ np.vstack((stim,np.ones(len(stim)))).T ]
    
    return inpts, true_choice, trial_start, response_inds
