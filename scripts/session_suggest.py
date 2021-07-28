# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:35:26 2021

@author: admin
"""
import sys
from uobrainflex.utils import djconnect
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema
from uobrainflex.pipeline import experimenter as experimenterSchema
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = 'no_subject'

djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()

all_sessions = djBehaviorSession.fetch(format='frame')
mouse_sessions = all_sessions.loc[all_sessions['subject_id'] == subject]

if mouse_sessions.empty:
        raise ValueError('You need to indicate a valid subject.')

last_session = mouse_sessions.iloc[-1,:]

if last_session['behavior_training_stage'][1] =='1':
    if all([last_session['licks_total']>2000, last_session['licks_left_ratio']>.4, last_session['licks_left_ratio']<.6]):
        next_session = 'S2'
    else:
        next_session = 'S1'
elif last_session['behavior_training_stage'][1] =='2':
    if all([last_session['hits_total']>250, last_session['hits_left_ratio']<.6, last_session['hits_left_ratio']>.4]):
        next_session = 'S3'
    elif all([last_session['licks_total']>2000, last_session['licks_left_ratio']>.2, last_session['licks_left_ratio']<.8]):
        next_session = 'S2'
    else:
        next_session = 'S1'
elif last_session['behavior_training_stage'][1] =='3':
    if all([last_session['hits_total']>250, last_session['hits_left_ratio']<.6, last_session['hits_left_ratio']>.4], 
           last_session['hit_rate_right']>.75, last_session['hit_rate_left']>.75):
        next_session = 'S4'
    if all([last_session['hits_total']>250, last_session['hits_left_ratio']<.6, last_session['hits_left_ratio']>.4]):
        next_session = 'S3'
    elif all([last_session['licks_total']>2000, last_session['licks_left_ratio']>.2, last_session['licks_left_ratio']<.8]):
        next_session = 'S2'
    else:
        next_session = 'S1'
else:
    next_session = last_session['behavior_training_stage'][0:2]
    
    
stage = np.array([])
for this_stage in mouse_sessions['behavior_training_stage']:
    stage = np.append(stage,int(this_stage[1]))
    
choice_ratio = mouse_sessions['choices_left_stim_ratio']
hit_rate = mouse_sessions['hit_rate_right']*(1-choice_ratio)+mouse_sessions['hit_rate_left']*choice_ratio

#plt.plot(stage/(max(stage)*2)-1/(max(stage)*2))
#plt.plot(mouse_sessions['hit_rate_right'])
#plt.plot(mouse_sessions['hit_rate_left'])

def stage_shift(y):
    return y*(max(stage)*4)+1
def stage_return(y):
    return y


fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(stage/(max(stage)*4)-1/(max(stage)*4))
ax.plot(mouse_sessions['licks_total']/max(mouse_sessions['licks_total']))
ax.plot(mouse_sessions['hits_total']/max(mouse_sessions['hits_total']))
ax.plot(mouse_sessions['choices_total']/max(mouse_sessions['choices_total']))
stage_axis = ax.secondary_yaxis('right',functions=(stage_shift,stage_return))
stage_axis.set_ylabel('Training Stage')
stage_axis.set_yticks(range(1,int(max(stage))+1))
plt.ylim([0, 1])
plt.xticks(ticks=range(len(mouse_sessions)),labels = [])
plt.xlabel('Sessions')
plt.ylabel('Total events (Norm)')
plt.legend(['Stage','Licks','Hits','Choices'])
plt.title(subject + ' events plot')

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(stage/(max(stage)*4)-1/(max(stage)*4))
ax.plot(mouse_sessions['licks_left_ratio'])
ax.plot(mouse_sessions['choices_left_stim_ratio'])
ax.plot(mouse_sessions['hits_left_ratio'])
stage_axis = ax.secondary_yaxis('right',functions=(stage_shift,stage_return))
stage_axis.set_ylabel('Training Stage')
stage_axis.set_yticks(range(1,int(max(stage))+1))
plt.ylim([0, 1])
plt.xticks(ticks=range(len(mouse_sessions)),labels = [])
plt.xlabel('Sessions')
plt.ylabel('Left ratio')
plt.legend(['Stage','Licks','Hits','Choices'])
plt.title(subject + ' bias plot')

print()
print('Based on the moset recent session in DJ, ' + subject + ' should next run on ' + next_session + ' next session.')
print()

