# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:58:29 2021

@author: Daniel Hulsey

Example utilizing uobrainflex packages to fit GLM_HMM model to behavioral sessions
"""

from uobrainflex.utils import djconnect
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema
from uobrainflex.pipeline import experimenter as experimenterSchema
from uobrainflex.nwb import loadbehavior as load
from uobrainflex.behavioranalysis import flex_hmm
import numpy as np


# two methods for getting nwbfilepaths are shown.
# 1. using a date range grabs the first file found on that date
# 2. using datajoint allows filtering of session based on parameters (e.g. training stage)
# comparing the two outputs is informative of potential pitfalls of the date range method - a stage 5 file is included, and a file is missed on Oct 2
# more extensive filtering of sessions based on DJ information could be useful (e.g. total hits, performance)

## option 1: getting behavior filepaths with date range
subject='BW041'; nwbfilepaths=[]
for fileID in range(20210929,20211013):
    this_session = load.get_file_path(subject,str(fileID))
    if this_session:
        nwbfilepaths.append(this_session)

## option 2: getting behavior file paths with datajoint
# import behavior session database from datajoint
djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()
all_sessions = djBehaviorSession.fetch(format='frame')

# filter sessions as desired
subject = 'BW041'
training_stage = 'S6'

mouse_sess = all_sessions.iloc[np.where(all_sessions['subject_id']==subject)[0]]
sess_idx = np.where(mouse_sess['behavior_training_stage']==training_stage)[0]
hmm_sessions = mouse_sess.iloc[sess_idx]

# get nwbfilepaths
nwbfilepaths = []
for fileID in hmm_sessions.index:
    this_session = load.get_file_path(subject,str(fileID[6:]))
    if this_session:
        nwbfilepaths.append(this_session)

# truncate selected sessions for this example
nwbfilepaths=nwbfilepaths[15:30]


## fitting GLM-HMM from list of filepaths
# load and format data. Getting behavior measures will significantly slow this down.
inpts, true_choices, hmm_trials = flex_hmm.compile_choice_hmm_data(nwbfilepaths, get_behavior_measures = False)

# generate glm-hmm model
# model fitting to determine number of states takes a while -- not shown here
# num_states = flex_hmm.choice_hmm_sate_fit(subject, inpts, true_choices, max_states = 4,save_folder='')
num_states = 3
hmm, hmm_trials = flex_hmm.choice_hmm_fit(subject, num_states, inpts, true_choices, hmm_trials)

## plot results
flex_hmm.plot_GLM_weights(subject, hmm)
flex_hmm.plot_state_occupancy(subject, hmm, true_choices, inpts)
flex_hmm.plot_session_posterior_probs(subject, hmm, true_choices, inpts)
flex_hmm.plot_state_posteriors_CDF(subject, hmm, inpts, true_choices)
flex_hmm.plot_state_psychometrics(subject, hmm, inpts, true_choices)
flex_hmm.plot_transition_matrix(subject, hmm)