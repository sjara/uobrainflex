# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:29:02 2024

@author: admin
"""

import sys
from pathlib import Path
from uobrainflex import config
from uobrainflex.nwb import loadbehavior as load
    
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
    return all([performance['licks_total']>1000,performance['hits_left_ratio']<.66, performance['hits_left_ratio']>.33])
        
def stage_2_pass(performance):
    return all([performance['choices_total']>100,performance['hit_rate_left']>.25, performance['hit_rate_right']>.25])

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

    if new_stage>int(performance['stage'][1])+1:
        new_stage = int(performance['stage'][1])+1
        
    if new_stage<int(performance['stage'][1])-1:
        new_stage = int(performance['stage'][1])-1 
        
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