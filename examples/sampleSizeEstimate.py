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
import pickle
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

obs_dim = 1
num_categories = 3
input_dim = 2
nIters = 1 #how many times to model select for a given mouse
num_sess = 1
stim_vals = [-.98,-.75,-.62,.62,.75,.98]
sessions_to_recover = []
for itr in range(nIters):
    sess_to_recov = flex_hmm.sampleSizeEstimation(subject, num_sess, inpts, true_choices, stim_vals, obs_dim, num_categories, input_dim)
    sessions_to_recover.append(sess_to_recov)

file_name = 'sampleSizeEst_' + subject +  '.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(sessions_to_recover, f)
