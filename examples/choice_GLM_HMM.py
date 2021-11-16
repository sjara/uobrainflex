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

# specify existing directory to save summary figures to. 
save_folder = '' # if save_folder == '', plots will be generated but not saved

# import behavior session database from datajoint. 
djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()
all_sessions = djBehaviorSession.fetch(format='frame')

# filter sessions as desired, this could also include a minimum hits/choices per session
subject = 'BW041'
training_stage = 'S6'
hmm_sessions = all_sessions.query("subject_id==@subject and behavior_training_stage==@training_stage")

# get nwbfilepaths
nwbfilepaths = []
for fileID in hmm_sessions.index:                                               #instead of using DJ to find fileIDs, fileID could be a range of dates
    this_session = load.get_file_path(subject,str(fileID[6:]))
    if this_session:
        nwbfilepaths.append(this_session)
        
nwbfilepaths = nwbfilepaths[-20:] # truncate sessions used for this example
## fitting GLM-HMM from list of filepaths`
# load and format data. Getting behavior measures will significantly slow this down.
inpts, true_choices, hmm_trials = flex_hmm.compile_choice_hmm_data(nwbfilepaths, get_behavior_measures = True)

## generate glm-hmm model
# model fitting to determine number of states takes a while -- not shown here
# num_states = flex_hmm.choice_hmm_sate_fit(subject, inpts, true_choices, max_states = 6,save_folder='')
num_states = 4
hmm = flex_hmm.choice_hmm_fit(subject, num_states, inpts, true_choices)
# include model outpit in trial data structures
posterior_probs, hmm_trials = flex_hmm.get_posterior_probs(hmm, true_choices, inpts, hmm_trials, occ_thresh = 0.8)


## plot results
flex_hmm.plot_GLM_weights(subject, hmm, save_folder)
flex_hmm.plot_state_occupancy(subject, hmm_trials, save_folder)
_ = flex_hmm.plot_dwell_times(subject, hmm_trials, save_folder)
flex_hmm.plot_state_posteriors_CDF(subject, posterior_probs, save_folder)
flex_hmm.plot_state_psychometrics(subject, hmm, inpts, true_choices, save_folder)
flex_hmm.plot_transition_matrix(subject, hmm, save_folder)

## plot relation between beahvior measures and state
flex_hmm.plot_session_summaries(subject, hmm_trials,nwbfilepaths,save_folder)
flex_hmm.plot_state_measure_histograms(subject, hmm_trials, save_folder)