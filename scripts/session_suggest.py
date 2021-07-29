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
    
if len(sys.argv)>2:
    num_sessions = sys.argv[2]
else:
    num_sessions = 10

djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()

all_sessions = djBehaviorSession.fetch(format='frame')
mouse_sessions = all_sessions.loc[all_sessions['subject_id'] == subject]
if len(mouse_sessions)>12:
    mouse_sessions = mouse_sessions[len(mouse_sessions)-num_sessions:]

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
    elif all([last_session['hits_total']>250, last_session['hits_left_ratio']<.6, last_session['hits_left_ratio']>.4]):
        next_session = 'S3'
    elif all([last_session['licks_total']>2000, last_session['licks_left_ratio']>.2, last_session['licks_left_ratio']<.8]):
        next_session = 'S2'
    else:
        next_session = 'S1'
else:
    next_session = last_session['behavior_training_stage'][0:2]

print('\nBased on the most recent session in DataJoint, ' + subject + ' should run on ' + next_session + ' next session.\n')    
    
stage = np.array([])
for this_stage in mouse_sessions['behavior_training_stage']:
    stage = np.append(stage,int(this_stage[1]))
    
choice_ratio = mouse_sessions['choices_left_stim_ratio']
hit_rate = mouse_sessions['hit_rate_right']*(1-choice_ratio)+mouse_sessions['hit_rate_left']*choice_ratio

def stage_shift(y):
    return y*(max(stage)*4)+1
def stage_return(y):
    return y

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(stage/(max(stage)*4)-1/(max(stage)*4))
ax.plot(mouse_sessions['licks_left_ratio'])
ax.plot(mouse_sessions['choices_left_stim_ratio'])
ax.plot(mouse_sessions['hits_left_ratio'])
ax.plot([0,len(mouse_sessions)],[.4,.4],'k--')
ax.plot([0,len(mouse_sessions)],[.6,.6],'k--')
ax.yaxis.set_ticks_position("right")
ax.yaxis.set_label_position("right")
stage_axis = ax.secondary_yaxis('left',functions=(stage_shift,stage_return))
stage_axis.set_ylabel('Training Stage')
stage_axis.set_yticks(range(1,int(max(stage))+1))
plt.ylim([0, 1])
plt.xlim([1,len(mouse_sessions)-1])
plt.xticks(ticks=range(len(mouse_sessions)),labels = [])
plt.xlabel('Sessions')
plt.ylabel('Left ratio')
plt.legend(['Stage','Licks','Choices','Hits'])
plt.title(subject + ' bias plot')
plt.show()


from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.7)

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par1,
                                    offset=(0, 0))

par1.axis["right"].toggle(all=True)


offset = 45
new_fixed_axis_2 = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis_2(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)

OFFSET = 90
new_fixed_axis_3 = par3.get_grid_helper().new_fixed_axis
par3.axis["right"] = new_fixed_axis_3(loc="right",
                                    axes=par3,
                                    offset=(OFFSET, 0))

par3.axis["right"].toggle(all=True)

host.set_xlim(0, len(mouse_sessions)-1)
host.set_ylim(1, 20)
host.set_yticks([1,2,3,4,5])

host.set_xlabel("Session")
host.set_ylabel("Stage")
par1.set_ylabel("Licks")
par2.set_ylabel("Choices")
par3.set_ylabel("Hits")

plt.xticks(ticks=range(len(mouse_sessions)),labels = [])
p1, = host.plot(stage, label="Stage")
p2, = par1.plot(mouse_sessions['licks_total'].values, label="Licks")
p3, = par2.plot(mouse_sessions['choices_total'].values, label="Choices")
p4, = par3.plot(mouse_sessions['hits_total'].values, label="Hits")
p5, = par1.plot([0,len(mouse_sessions)],[1000, 1000],'--', label="Licks")
p6, = par3.plot([0,len(mouse_sessions)],[200, 200],'--', label="Hits")

p5.set_color(p2.get_color())
p6.set_color(p4.get_color())

par1.set_ylim(0, max(mouse_sessions['licks_total']))
par2.set_ylim(0, max(mouse_sessions['choices_total']))
par2.set_ylim(0, max(mouse_sessions['hits_total']))

#host.legend()
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())
par3.axis["right"].label.set_color(p4.get_color())

plt.title(subject + ' events plot')

plt.draw()
plt.show()
