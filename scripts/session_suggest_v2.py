# -*- coding: utf-8 -*-
"""
Read most recently uploaded .nwb behavior record for input subjects and suggest a training stage for their next training session.

run from command line and input full subject IDs separated by spaces.
"""

import sys
from pathlib import Path
from uobrainflex.nwb import loadbehavior as load
from uobrainflex import config

def get_performance(nwb_filepath):
    trial_data, trial_labels = load.load_trial_data(str(nwb_filepath))
    _,events = load.load_behavior_events(str(nwb_filepath))
    
    left_licks = len(events['licks_left'])
    right_licks = len(events['licks_right'])
    left_choices = len(trial_data.query('target_port==0 and outcome<3'))
    if left_choices>0:
        left_hr = len(trial_data.query('target_port==0 and outcome==1'))/left_choices
    else:
        left_hr = 0
    right_choices = len(trial_data.query('target_port==1 and outcome<3'))
    if  right_choices>0:
        right_hr = len(trial_data.query('target_port==1 and outcome==1'))/right_choices
    else:
        right_hr = 0
    left_nr = len(trial_data.query('target_port==0 and outcome==16'))
    total_nr = len(trial_data.query('outcome==16'))
    left_nr_ratio = left_nr/total_nr
    hits_total = len(trial_data.query('outcome==1'))
    if hits_total > 0:
        hits_left_ratio = len(trial_data.query('target_port==0 and outcome==1'))/hits_total     
    else:
        hits_left_ratio = 0
    
    performance={}
    performance['stage'] = trial_data['stage'].iloc[0]
    performance['licks_total'] = left_licks + right_licks
    performance['licks_left_ratio'] = round(left_licks / (left_licks + right_licks),2)
    performance['choices_total'] = len(trial_data.query('outcome<3'))
    performance['hits_total'] = len(trial_data.query('outcome==1'))
    performance['hits_left_ratio'] = round(hits_left_ratio,2)
    performance['hit_rate_right'] = round(right_hr,2)
    performance['hit_rate_left'] = round(left_hr,2)
    performance['no_response_total'] = total_nr
    performance['no_response_left_ratio'] = round(left_nr_ratio,2)
    
    return performance
    
def stage_1_pass(performance):
    return all([performance['licks_total']>2000,performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33])
        
def stage_2_pass(performance):
    return all([performance['choices_total']>150,performance['hit_rate_left']>.25, performance['hit_rate_right']>.25])

def stage_3_pass(performance):
    return all([performance['hits_total']>150, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33, 
               performance['hit_rate_right']>.6, performance['hit_rate_left']>.6, 
               performance['no_response_total']<40 or (performance['no_response_left_ratio']<.7 and performance['no_response_left_ratio']>.3)])

def stage_4_pass(performance):
    return all([performance['hits_total']>150, performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33, 
               performance['hit_rate_right']>.6, performance['hit_rate_left']>.6])

def generate_suggestion(performance):
    if stage_1_pass(performance):
        if stage_2_pass(performance):
            if stage_3_pass(performance):
                if stage_4_pass(performance):
                    new_stage = 5
                else:
                    new_stage = 4
            else: 
                new_stage = 3
        else:
            new_stage = 2
    else:
        new_stage = 1

    return new_stage
        
if len(sys.argv)>1:
    subjects = sys.argv[1:]
else:
    subjects = 'no_subject'

p = Path(config['BEHAVIOR']['cloud_storage_path'])

for subject in subjects:
    files = list(p.joinpath(subject).glob(subject +'*.nwb'))
    performance = get_performance(files[-1]) 
    new_stage = generate_suggestion(performance)
    
    if new_stage>int(performance['stage'][1])+1:
        new_stage = int(performance['stage'][1])+1
    
    print('\n')
    print(subject + ' last ran on stage ' + str(performance['stage']) + '. Consider stage S' + str(new_stage) + ' next session:')
    print(performance)

    
    
# ###################
# import matplitlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle

# p = Path(r'Z:\behavior')
# subject = 'BW100'
# files = list(p.joinpath(subject).glob(subject +'*.nwb'))
# n_sessions = 3
# plt.figure(figsize=[20,2*n_sessions])    
# ax=[]
# for i,file in enumerate(files[-n_sessions:]):
#     trial_data, trial_labels = load.load_trial_data(str(file))
#     _,events = load.load_behavior_events(str(file))
    
#     ax.append(plt.subplot(n_sessions,1,i+1))
#     hand=[]
#     lg=[]
    
#     #plot hmm posterior probabilities
#     t_zero = trial_data['start_time'].iloc[0]
#     t = trial_data['start_time'].values-t_zero
       
#     ax[-1].add_patch(Rectangle((t[0],1.15), t[-1], .095,color=[.7,.7,.7],alpha=.3,clip_on=False,edgecolor = None,linewidth=0))
#     ax[-1].add_patch(Rectangle((t[0],1.05), t[-1], .095,color=[.7,.7,.7],alpha=.6,clip_on=False,edgecolor = None,linewidth=0))
    
    
#     #plot individual trial outcomes/stim values
#     for i in range(len(trial_data)):
#         choice = trial_data.iloc[i]['choice']
#         outcome = trial_data.iloc[i]['outcome']
#         inpt = trial_data.iloc[i]['inpt']
#         if choice == 2:
#             choice=1.2
#             if inpt<0:
#                 choice=1.225
#             else:
#                 choice=1.175
#         elif choice ==0:
#             choice = 1.3
#         elif choice == 1:
#             choice = 1.1     
            
#         if outcome==16:
#             outcome=3        
#                 ###
#         if inpt<0:
#             m='k'
#         else:
#             m='r'        
#         if abs(inpt) == .2:
#             msize = 15
#         elif abs(inpt) == .6:
#             msize = 25
#         else:
#             msize= 35 
                
#         ax[-1].scatter(t[i],1.4-outcome/10+np.random.randn()/85,color=m,s=msize,clip_on=False,edgecolor = None,linewidth=0,zorder=10)
                
    
#     ## plaen E legends
#     ax[-1].set_ylim([1,1.4])
#     ax[-1].set_yticks([])
#     ax[-1].set_xlim([-10,t[-1]])
#     for spine in ['top','right','bottom','left']:
#         ax[-1].spines[spine].set_visible(False)
#     ax[-1].set_xticks([])
#     ax[-1].text(-20,1.1,'No resp.',ha='right',va='center')
#     ax[-1].text(-20,1.2,'Error',ha='right',va='center')
#     ax[-1].text(-20,1.3,'Hit',ha='right',va='center')