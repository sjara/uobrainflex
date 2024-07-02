# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:29:02 2024

@author: admin
"""

import numpy as np
import ssm
import sys
from pathlib import Path
from uobrainflex import config
from uobrainflex.nwb import loadbehavior as load
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from uobrainflex.behavioranalysis import flex_hmm
    
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
         
def generate_hmm():
    trans_mat_6 = np.full([6,6],.015)
    for i in range(6):
        trans_mat_6[i,i] = .925
    
    ws = np.array([[[-1.3,4.5], [1.3,4.5]],
                    [[-.8,-3],  [.8,-3]],
                    [[-2,2.5],  [.8,-.5]],
                    [[-2.8,1],  [.3,-1.5]],
                    [[-.8,-.5], [2,2.5]],
                    [[-.3,-1.5],[2.8,1]]])
    
    model_mouse_hmm =  ssm.HMM(6, 1, 2, observations="input_driven_obs", 
                       observation_kwargs=dict(C=3), transitions="standard")
    
    model_mouse_hmm.params = (model_mouse_hmm.params[0],np.log(trans_mat_6)[None,:,:],ws)
    return model_mouse_hmm

def get_state_occupancy(hmm,trial_data):
    
    true_choices = trial_data['choice'].to_numpy()
    stim_val = trial_data['inpt'].to_numpy()
    inpt = np.vstack([stim_val,np.ones(len(stim_val))]).T
    
    posterior_probs = hmm.expected_states(data=true_choices[:,None], input=inpt)[0]
    n,bins = np.histogram(np.argmax(posterior_probs,axis=1),bins=np.arange(-.5,6,1))
    percents = np.round(n/sum(n),2)*100
    return percents

def plot_outcomes_and_states(all_trials,posterior_probs):   
   
    n_sessions = len(all_trials)
    plt.figure(figsize=[20,2*n_sessions])    
    ax=[]
    for ii, trial_data in enumerate(all_trials):
        ax.append(plt.subplot(n_sessions,1,ii+1))        
                
        #plot hmm posterior probabilities
        t_zero = trial_data['start_time'].iloc[0]
        t = trial_data['start_time'].values-t_zero
           
        ax[-1].add_patch(Rectangle((t[0],1.15), t[-1], .095,color=[.7,.7,.7],alpha=.3,clip_on=False,edgecolor = None,linewidth=0))
        ax[-1].add_patch(Rectangle((t[0],1.05), t[-1], .095,color=[.7,.7,.7],alpha=.6,clip_on=False,edgecolor = None,linewidth=0))
        
        
        #plot individual trial outcomes/stim values
        for i in range(len(trial_data)):
            choice = trial_data.iloc[i]['choice']
            outcome = trial_data.iloc[i]['outcome']
            inpt = trial_data.iloc[i]['inpt']
            if choice == 2:
                choice=1.2
                if inpt<0:
                    choice=1.225
                else:
                    choice=1.175
            elif choice ==0:
                choice = 1.3
            elif choice == 1:
                choice = 1.1     
                
            if outcome==16:
                outcome=3        
                    ###
            if inpt<0:
                m='k'
            else:
                m='r'        
            if any([abs(inpt) == .2,abs(inpt) == .43]):
                msize = 15
            elif any([abs(inpt) == .6,abs(inpt) == .626]):
                msize = 25
            else:
                msize= 35 
                    
            ax[-1].scatter(t[i],1.4-outcome/10+np.random.randn()/85,color=m,s=msize,clip_on=False,edgecolor = None,linewidth=0,zorder=10)
                    
        ax[-1].set_ylim([1,1.4])
        ax[-1].set_yticks([])
        ax[-1].set_xlim([-10,t[-1]])
        for spine in ['top','right','bottom','left']:
            ax[-1].spines[spine].set_visible(False)
        ax[-1].set_xticks([])
        ax[-1].text(-20,1.1,'No resp.',ha='right',va='center')
        ax[-1].text(-20,1.2,'Error',ha='right',va='center')
        ax[-1].text(-20,1.3,'Hit',ha='right',va='center')
    
        ax[-1].plot(t, posterior_probs[ii]*.3+1.05)
        ax[-1].plot([t[0],t[-1]],[.8*.3+1.05,.8*.3+1.05],'k--')
        ax[-1].yaxis.set_label_position("right")
        plt.ylabel(trial_data.iloc[0]['stage'])
        plt.title(files[ii])
    plt.show()

if len(sys.argv)>1:
    subjects = sys.argv[1:]
else:
    subjects = 'no_subject'

p = Path(config['BEHAVIOR']['cloud_storage_path'])
hmm = generate_hmm()

if len(subjects)>1:
    for subject in subjects:
        files = list(p.joinpath(subject).glob(subject +'*.nwb'))
        performance = get_performance(files[-1]) 
        new_stage = generate_suggestion(performance)
        
        if new_stage>int(performance['stage'][1])+1:
            new_stage = int(performance['stage'][1])+1
        
        print('\n')
        print(subject + ' last ran on stage ' + str(performance['stage']) + '. Consider stage S' + str(new_stage) + ' next session:')
        print(performance)
        
else:

    n_sessions = 3
    files = list(p.joinpath(subjects[0]).glob(subjects[0] +'*.nwb'))[-n_sessions:]
    performance = get_performance(files[-1]) 
    new_stage = generate_suggestion(performance)
    inpts=[]
    true_choices=[]
    all_trials=[]
    for i,file in enumerate(files[-n_sessions:]):
        trial_data, trial_labels = load.load_trial_data(str(file))
        inpt, true_choice, trials = flex_hmm.format_choice_behavior_hmm(trial_data, trial_labels)
        true_choices.append(true_choice)
        inpts.append(inpt[0])
        all_trials.append(trials)
        
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
                   for data, inpt
                   in zip(true_choices, inpts)]

    plot_outcomes_and_states(all_trials,posterior_probs)
    print('\n')
    print(subjects[0] + ' last ran on stage ' + str(performance['stage']) + '. Consider stage S' + str(new_stage) + ' next session:')
    print(performance)

