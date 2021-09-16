# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 08:42:06 2021

@author: admin
"""

from uobrainflex.nwb import loadbehavior as load
import numpy as np

## for testing 
# import pynwb
# import matplotlib.pyplot as plt
# nwbfilepath = load.get_file_path('BW031','20210806')
# ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
# nwbFileObj = ioObj.read()  

def lick_analysis(nwbFileObj):
    behavior_events, behavior_events_index = load.read_behavior_events(nwbFileObj,speriod=0.001,events=[])
    licks_left = len(behavior_events['licks_left'])
    licks_right = len(behavior_events['licks_right'])
    licks_total = licks_left + licks_right
    if licks_total == 0:
        licks_left_ratio = np.nan
    else:
        licks_left_ratio = licks_left/licks_total
        
    return licks_left_ratio, licks_total
    
def response_analysis(trial_data, trial_dict):    
    distractor_trials = trial_data.iloc[np.where(trial_data['stimulus_type'] == trial_dict['stimulus_type']['distractor'])[0],:]
    fa_trials = distractor_trials.drop(index = distractor_trials.index[distractor_trials['outcome']==trial_dict['outcome']['correct_reject']].tolist())
    if len(distractor_trials) > 0:
        fa_rate = len(fa_trials)/len(distractor_trials)
    else:
        fa_rate = np.nan
    target_trials = trial_data.iloc[np.where(trial_data['stimulus_type'] != trial_dict['stimulus_type']['distractor'])[0],:]   
    response_trials = target_trials.drop(index = target_trials.index[target_trials['outcome']==trial_dict['outcome']['incorrect_reject']].tolist())   
    
    if len(target_trials) > 0:
        response_rate = len(response_trials)/len(target_trials)
    else:
        response_rate = np.nan

    return response_rate, fa_rate

def choice_analysis(trial_data, trial_dict):
    target_trials = trial_data.iloc[np.where(trial_data['stimulus_type'] != trial_dict['stimulus_type']['distractor'])[0],:]   
    response_trials = target_trials.drop(index = target_trials.index[target_trials['outcome']==trial_dict['outcome']['incorrect_reject']].tolist())   
    
    choices_total = len (response_trials)
    left_trial_choices = len(np.where(response_trials.target_port == trial_dict['target_port']['left'])[0])
    
    if choices_total == 0:
        choices_left_trial_ratio = np.nan
    else:
        choices_left_trial_ratio = left_trial_choices/choices_total

    return choices_total, choices_left_trial_ratio

def hit_analysis(trial_data, trial_dict):
    target_trials = trial_data.iloc[np.where(trial_data['stimulus_type'] != trial_dict['stimulus_type']['distractor'])[0],:]   
    response_trials = target_trials.drop(index = target_trials.index[target_trials['outcome']==trial_dict['outcome']['incorrect_reject']].tolist())   
    
    left_trials = np.where(response_trials['target_port'] == trial_dict['target_port']['left'])[0]
    right_trials = np.where(response_trials['target_port'] == trial_dict['target_port']['right'])[0]
    hits = np.where(response_trials.outcome == trial_dict['outcome']['hit'])[0]

    left_hits = len(np.intersect1d(left_trials,hits))
    right_hits = len(np.intersect1d(right_trials,hits))

    hits_total = left_hits+right_hits
    if hits_total == 0:
        hits_left_ratio = np.nan
    else:
        hits_left_ratio = left_hits/hits_total
        
    left_hits = len(np.intersect1d(left_trials,hits))
    right_hits = len(np.intersect1d(right_trials,hits))
    if len(left_trials) == 0:
        hit_rate_left = np.nan
    else:
        hit_rate_left = left_hits/len(left_trials)
    if len(right_trials) ==0:
        hit_rate_right = np.nan
    else:
        hit_rate_right = right_hits/len(right_trials)
    if all([len(right_trials), len(left_trials)]):
        hit_rate = len(hits)/(len(left_trials)+len(right_trials))
    else:
        hit_rate = np.nan
        
    return hits_total, hit_rate, hits_left_ratio, hit_rate_left, hit_rate_right


def get_summary(nwbFileObj):
    """
    Return performance metrics for a specified nwb file.
    Args:
        nwbfilepath (string): full path to an NWB file.
    Returns:
        performance (dict): dictionary of performance calculations
    """
    
## licks
    licks_left_ratio, licks_total = lick_analysis(nwbFileObj)

## choices
    trial_data, trial_dict = load.read_trial_data(nwbFileObj)
    
    response_rate, fa_rate = response_analysis(trial_data, trial_dict)
    choices_total, choices_left_trial_ratio = choice_analysis(trial_data, trial_dict)
    hits_total, hit_rate, hits_left_ratio, hit_rate_left, hit_rate_right = hit_analysis(trial_data, trial_dict)

## compile performance summary
    performance_summary = {}
    performance_summary['hits_total'] = hits_total
    performance_summary['hits_left_ratio'] = hits_left_ratio
    performance_summary['licks_total'] = licks_total
    performance_summary['licks_left_ratio'] = licks_left_ratio
    performance_summary['choices_total'] = choices_total
    performance_summary['choices_left_stim_ratio'] = choices_left_trial_ratio
    performance_summary['hit_rate_left'] = hit_rate_left
    performance_summary['hit_rate_right'] = hit_rate_right
    performance_summary['hit_rate_total'] = hit_rate
    performance_summary['false_alarm_rate'] = fa_rate
    performance_summary['rig'] = nwbFileObj.lab_meta_data['metadata'].rig
    performance_summary['stage'] = nwbFileObj.lab_meta_data['metadata'].training_stage

    return performance_summary

def response_times(nwbFileObj):
    trial_data, trial_dict = load.read_trial_data(nwbFileObj)
    behavior_events, behavior_events_index = load.read_behavior_events(nwbFileObj,speriod=0.001,events=[])
    
    behavior_events['licks_left']
    l_lick = behavior_events_index['licks_left']*0.001
    r_lick = behavior_events_index['licks_right']*0.001
    
    licks = np.append(r_lick,l_lick)
    licks.sort()

    trial_data['stimulus_time']
    
    response_time=([])
    for t in trial_data['stimulus_time']:
        ind = np.where(licks - t > 0)[0]
        if len(ind) >0:
            rt = licks[ind[0]]-t
            if rt < 2.5:
                response_time = np.append(response_time,rt)
            else:
                response_time = np.append(response_time,np.nan)
        else:
            response_time = np.append(response_time, np.nan)
    return response_time
    
# import pynwb
# import matplotlib.pyplot as plt
# nwbfilepath = load.get_file_path('BW016','20210421')
# nwbfilepath = load.get_file_path('BW016','20210422')
# nwbfilepath = load.get_file_path('BW016','20210408')
# ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
# nwbFileObj = ioObj.read()  

# response_time = response_times(nwbFileObj)
# plt.hist(response_time,np.arange(0,2,.01))