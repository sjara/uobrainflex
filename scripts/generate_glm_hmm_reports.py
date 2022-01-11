# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:58:29 2021

@author: Daniel Hulsey

Example utilizing uobrainflex packages to fit GLM_HMM model to behavioral sessions
"""

import os
from uobrainflex.utils import djconnect
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema
from uobrainflex.pipeline import experimenter as experimenterSchema
from uobrainflex.nwb import loadbehavior as load
from uobrainflex.behavioranalysis import flex_hmm
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


save_dialog = 'select folder to save hmm reports'
root = tk.Tk()
root.withdraw()
save_folder = filedialog.askdirectory(title = save_dialog)

subject = input('enter subject name \n')

if not(os.path.exists(save_folder)):
    os.mkdir(save_folder)
save_folder =  save_folder + subject +'\\' 
if not(os.path.exists(save_folder)):
    os.mkdir(save_folder)
save_folder = save_folder + datetime.today().strftime('%Y-%m-%d') +'\\'
if not(os.path.exists(save_folder)):
    os.mkdir(save_folder)    

base_folder = save_folder

# import behavior session database from datajoint. 
djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()
all_sessions = djBehaviorSession.fetch(format='frame')

# filter sessions as desired, this could also include any datajoint columns
training_stage = ['S5','S6']
min_choices = 100
date = '2020-02-28'
hmm_sessions = all_sessions.query("subject_id==@subject and choices_total>@min_choices and behavior_session_start_time>@date and behavior_training_stage==@training_stage")

# get nwbfilepaths
nwbfilepaths = []
for fileID in hmm_sessions.index:
    this_session = load.get_file_path(subject,str(fileID[6:]))
    if this_session:
        nwbfilepaths.append(this_session)
        
# load and format data
inpts, true_choices, hmm_trials = flex_hmm.compile_choice_hmm_data(nwbfilepaths, get_behavior_measures = True)

# model fitting to determine appropriate number of states
num_states = flex_hmm.choice_hmm_sate_fit(subject, inpts, true_choices, max_states = 6,save_folder=base_folder)

 # generate reports for models with 3-6 states 
for num_states in range(3,7):
    print('fitting model with ' + str(num_states) + ' states')
    save_folder = base_folder
    if save_folder != '':
        save_folder = save_folder + str(num_states) + '_states\\'
        if not(os.path.exists(save_folder)):
            os.mkdir(save_folder)
        line_folder = save_folder + 'line summaries\\'
        patch_folder = save_folder + 'patch summaries\\'
        if not(os.path.exists(line_folder)):
            os.mkdir(line_folder)
            os.mkdir(patch_folder)
    
    # fit model and append trial data with its output    
    hmm = flex_hmm.choice_hmm_fit(subject, num_states, inpts, true_choices)
    posterior_probs, hmm_trials = flex_hmm.get_posterior_probs(hmm, true_choices, inpts, hmm_trials, occ_thresh = 0.8)
    
    # permute states to set high performing first and remap hmm_trials
    flex_hmm.permute_hmm(hmm,hmm_trials)
    posterior_probs, hmm_trials = flex_hmm.get_posterior_probs(hmm, true_choices, inpts, hmm_trials, occ_thresh = 0.8)
    
    print('plotting resutls of model with ' + str(num_states) + ' states')
    ## plot results
    flex_hmm.plot_state_psychometrics(subject, hmm_trials, save_folder)
    flex_hmm.plot_whisk_by_state(subject, hmm_trials, save_folder)
    flex_hmm.plot_locomotion_by_state(subject, hmm_trials, save_folder)
    flex_hmm.plot_GLM_weights(subject, hmm, save_folder)
    flex_hmm.plot_state_occupancy(subject, hmm_trials, save_folder)
    dwell_times = flex_hmm.plot_dwell_times(subject, hmm_trials, save_folder)
    flex_hmm.plot_state_posteriors_CDF(subject, posterior_probs, save_folder)
    flex_hmm.plot_transition_matrix(subject, hmm, save_folder)
    
    ## plot relation between beahvior measures and state
    flex_hmm.plot_measures_by_state(subject, hmm_trials, save_folder)
    flex_hmm.plot_session_summaries(subject, hmm_trials, nwbfilepaths,line_folder)
    # flex_hmm.plot_session_summaries_patch(subject, hmm_trials, dwell_times,nwbfilepaths,patch_folder)
    flex_hmm.report_cover(subject, nwbfilepaths[0],save_folder)
    
flex_hmm.generate_report(subject, base_folder)

