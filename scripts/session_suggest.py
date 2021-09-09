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


def generate_suggestion(performance):
    '''
    Suggest a session based on performance measures
    Args:
        performance (dict): dictionary containing stage, lick (total/ratio), hit (total/ratio), and hit rate (L/R) information
    Returns:
        next_session (str): String of stage suggestion
    '''
    if performance['stage'][1] == '1':
        if all([performance['licks_total']>1000, performance['licks_left_ratio']>.33, performance['licks_left_ratio']<.66]):
            next_session = 'S2'
        else:
            next_session = 'S1'
    elif performance['stage'][1] == '2':
        if all([performance['hits_total']>200,performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33]):
            next_session = 'S3'
        elif any([all([performance['licks_total']>1000, performance['licks_left_ratio']>.3, performance['licks_left_ratio']<.7]),
                 all([performance['hits_total']*performance['hits_left_ratio']>25, performance['hits_total']*(1-performance['hits_left_ratio'])>25])]):
            next_session = 'S2'
        else:
            next_session = 'S1'
    elif performance['stage'][1] == '3':
        if all([performance['hits_total']>200, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33, 
               performance['hit_rate_right']>.75, performance['hit_rate_left']>.75]):
            next_session = 'S4'
        elif all([performance['hits_total']>100, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33]):
            next_session = 'S3'
        elif any([all([performance['licks_total']>1000, performance['licks_left_ratio']>.3, performance['licks_left_ratio']<.7]),
                 all([performance['hits_total']*performance['hits_left_ratio']>25, performance['hits_total']*(1-performance['hits_left_ratio'])>25])]):
            next_session = 'S2'
        else:
            next_session = 'S1'
    elif performance['stage'][1] =='4':
        if all([performance['hits_total']>200, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33, 
               performance['hit_rate_right']>.75, performance['hit_rate_left']>.75]):
            next_session = 'S5'
        elif all([performance['hits_total']>100, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33, 
               performance['hit_rate_right']>.75, performance['hit_rate_left']>.75]):
            next_session = 'S4'
        elif all([performance['hits_total']>100, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33]):
            next_session = 'S3'
        elif any([all([performance['licks_total']>1000, performance['licks_left_ratio']>.3, performance['licks_left_ratio']<.7]),
                 all([performance['hits_total']*performance['hits_left_ratio']>25, performance['hits_total']*(1-performance['hits_left_ratio'])>25])]):
            next_session = 'S2'
        else:
            next_session = 'S1'
    else:
        next_session = last_session['behavior_training_stage'][0:2]
    return next_session
        
if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = 'no_subject'
    
if len(sys.argv)>2:
    plotStr = sys.argv[2]
else:
    plotStr = 'n'    
    
if len(sys.argv)>3:
    num_sessions = int(sys.argv[3])
else:
    num_sessions = 10

plot_behavior = plotStr.lower()=='y'

djSubject = subjectSchema.Subject()
djExperimenter = experimenterSchema.Experimenter()
djBehaviorSession = acquisitionSchema.BehaviorSession()

all_sessions = djBehaviorSession.fetch(format='frame')
mouse_sessions = all_sessions.loc[all_sessions['subject_id'] == subject]
if len(mouse_sessions)>num_sessions:
    mouse_sessions = mouse_sessions[len(mouse_sessions)-num_sessions:]

if mouse_sessions.empty:
        raise ValueError('You need to indicate a valid subject.')

last_session = mouse_sessions.iloc[-1,:]
last_2session = mouse_sessions.iloc[len(mouse_sessions)-2:]


choices_total =  last_session['choices_total']
choices_left_ratio =  last_session['choices_left_stim_ratio']
choice_right_stim = choices_total*(1-choices_left_ratio)
choice_left_stim = choices_total*choices_left_ratio


performance={}
performance['stage'] = last_session['behavior_training_stage']
performance['licks_total'] = last_session['licks_total']
performance['licks_left_ratio'] = last_session['licks_left_ratio']
performance['hits_total'] = last_session['hits_total']
performance['hits_left_ratio'] = last_session['hits_left_ratio']
performance['hit_rate_right'] = last_session['hit_rate_right']
performance['hit_rate_left'] = last_session['hit_rate_left']

next_session = generate_suggestion(performance)
print('\nBased on the most recent session in DataJoint, ' + subject + ' should run on ' + next_session + ' next session.\n')    


# if the last two sessions occured on the same day and have the same stage, then combine them for a second suggestion
if all([last_2session['behavior_training_stage'][0] == last_2session['behavior_training_stage'][1],
        last_2session['behavior_filename'][0][0:23] == last_2session['behavior_filename'][1][0:23]]): 
    licks_total = last_2session['licks_total'][0]
    hits_total = last_2session['hits_total'][0]
    licks_left_ratio = last_2session['licks_left_ratio'][0]
    hits_left_ratio  = last_2session['hits_left_ratio'][0] 
    hit_rate_right = last_2session['hit_rate_right'][0]
    hit_rate_left  = last_2session['hit_rate_left'][0]
    choices_total_2 =  last_2session['choices_total'][0]
    choices_left_ratio_2 =  last_2session['choices_left_stim_ratio'][0]
    choice_right_stim_2 = choices_total_2*(1-choices_left_ratio_2)
    choice_left_stim_2 = choices_total_2*choices_left_ratio_2
    
    performance['hit_rate_right'] = performance['hit_rate_right']*(choice_right_stim/(choice_right_stim+choice_right_stim_2)) + hit_rate_right*(choice_right_stim_2/(choice_right_stim+choice_right_stim_2))
    performance['hit_rate_left'] = performance['hit_rate_left']*(choice_left_stim/(choice_left_stim+choice_left_stim_2)) + hit_rate_left*(choice_left_stim_2/(choice_left_stim+choice_left_stim_2)) 
    performance['licks_left_ratio'] = performance['licks_left_ratio']*(performance['licks_total']/(licks_total + performance['licks_total'])) + licks_left_ratio*(licks_total/(performance['licks_total'] + licks_total))
    performance['licks_total'] = performance['licks_total'] + licks_total
    performance['hits_left_ratio'] = performance['hits_left_ratio']*(performance['hits_total']/(performance['hits_total'] + hits_total)) + hits_left_ratio*(hits_total/(performance['hits_total'] + hits_total))
    performance['hits_total'] = performance['hits_total'] + hits_total
        
    next_session = generate_suggestion(performance)
    print('Based on the last 2 sessons in DataJoint on stage ' + performance['stage'] + ' on the same day, ' + subject + ' should run on ' + next_session + ' next session.\n')       

if plot_behavior:

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    
    stage = np.array([])
    for this_stage in mouse_sessions['behavior_training_stage']:
        stage = np.append(stage,int(this_stage[1]))
        
        
    
    ## bias plot
    def stage_shift(y):
        return y*(max(stage)*4)+1
    def stage_return(y):
        return y
    
    choice_ratio = mouse_sessions['choices_left_stim_ratio']
    hit_rate = mouse_sessions['hit_rate_right']*(1-choice_ratio)+mouse_sessions['hit_rate_left']*choice_ratio

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
    
    ## events plot
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
