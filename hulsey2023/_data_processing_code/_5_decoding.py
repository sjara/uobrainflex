# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:55:59 2022

@author: admin
"""

# load hmm_trials and use cross-validated SVM classifiers to predcit states from arousal & movement measures


import os
import numpy as np
import glob
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
import time
import ray
from scipy import stats


start_time = time.time()


base_folder = 'E:\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

measures_to_use = ['post_hoc_pupil_diameter', 'post_hoc_pupil_std10',
                                    'face_energy',  'face_std_10',
                                    'running_speed', 'running_std10',
                                    'movement','movement_std'] 

## shuffle measures to determine accuracy using subsets of the data
shuffle_schedule=[[6,7],[2,3,4,5], # movement index; face + wheel
                  [2,3,4,5,6,7],[0,1,4,5,6,7],[0,1,2,3,6,7],[1,2,3,4,5], #keep: pupil only,face only,wheel only,movement idx only
                  [0,1,6,7],[2,3,6,7],[4,5,6,7],#shuffle pupil+move, face+move,wheel+move
                  [1,3,5,6,7],[0,2,4,6,7], #pupil,face,wheel: val only, var only
                  [1,2,3,4,5,6],[0,2,3,4,5,7],#pupil,move: val only, var only
                  [],[0,1,2,3,4,5,6,7] ] # shuffle all
                  

n_possible_states=6
nKfold=5
kernel = 'rbf'
n_run=1000
SVM_acc=np.full([len(hmm_trials_paths),n_possible_states,n_run,len(shuffle_schedule),nKfold],np.nan)
SVM = svm.SVC()
SVM_models=np.full([len(hmm_trials_paths),n_possible_states,n_run,len(shuffle_schedule),nKfold],0)

#parallelize the classifier fitting
@ray.remote
def fit_and_score_SVM(train_measures,train_states,test_measures,test_states,kernel='rbf'):
    SVM = svm.SVC(kernel=kernel)
    SVM.fit(train_measures,train_states.ravel())  
    accuracy = SVM.score(test_measures,test_states)
    return [SVM,accuracy]

#############

ray.init()
subjects=[]
for m in range(len(hmm_trials_paths)):
    # mouse_SVMS = np.full([6,n_run,len(shuffle_schedule),5],0)
    mouse_SVM_acc = np.full([n_possible_states,n_run,len(shuffle_schedule),nKfold],np.nan)
    
    print('mouse ' + str(m+1) + ' of ' + str(len(hmm_trials_paths)))
    subject = os.path.basename(hmm_trials_paths[m])[:5]
    print(subject)
    subjects.append(subject)
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
    

    #grab all trials from sessions with optimal state
    all_trials = pd.DataFrame()
    for i,trials in enumerate(hmm_trials):
        if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'post_hoc_pupil_diameter'
        elif all([len(trials.query('hmm_state==0'))>10,np.isin('pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'pupil_diameter'

    
    # drop trials with indeterminate state
    state_trials = all_trials.query('hmm_state==hmm_state')
    for measure in measures_to_use[:6]:
        state_trials = state_trials.query(measure +'=='+measure)
    
    #make movement index
    state_trials['movement'] = stats.zscore(state_trials['face_energy']) + stats.zscore(state_trials['running_speed'])
    state_trials['movement_std'] = stats.zscore(state_trials['face_std_10']) + stats.zscore(state_trials['running_std10'])
    
    

    states_to_use = np.unique(state_trials['hmm_state'])
    states_to_use = states_to_use[1:]
    for state_iter, this_state in enumerate(states_to_use):
        print('  state ' + str(state_iter+1) + ' of ' + str(len(states_to_use)))
        opt_trials = state_trials.query('hmm_state==0')
        this_state_trials = state_trials.query('hmm_state==@this_state')
        
        opt_trials = opt_trials.iloc[np.random.permutation(range(len(opt_trials)))]   
        this_state_trials = this_state_trials.iloc[np.random.permutation(range(len(this_state_trials)))]   
        
        # take up to 1500 randomly sampled trials
        if len(opt_trials)>1500:
            opt_trials=opt_trials.iloc[:1500]
        if len(this_state_trials)>1500:
            this_state_trials=this_state_trials.iloc[:1500]
        
        
        if len(opt_trials)>len(this_state_trials):
            opt_trials = opt_trials.iloc[:len(this_state_trials)]
        elif len(opt_trials)<len(this_state_trials):
            this_state_trials = this_state_trials.iloc[:len(opt_trials)]



        states = np.concatenate([opt_trials['hmm_state'].values,this_state_trials['hmm_state'].values])
        measures = np.concatenate([opt_trials[measures_to_use].values,this_state_trials[measures_to_use].values])
        
        #normalize data 
        min_max_scaler = preprocessing.MinMaxScaler()
        measures_scaled = min_max_scaler.fit_transform(measures)
       
        kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)  
        # not not enough trials for a state skip it
        if len(states)>30:
            for ii, (train_index, test_index) in enumerate(kf.split(states)):
                print(f"    kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
                for run in range(n_run):
                    if run%(n_run/5) ==0:
                        print(f"      run {run} of " + str(n_run))
                    processes=[]
                    
                    #randomly permute training set data according to shuffle_schedule
                    permutation = np.random.permutation(len(train_index))
                    for condition in range(len(shuffle_schedule)):
                        train_measures = np.array(measures_scaled[train_index])
                        train_states = np.array(states[train_index])
                        
                        test_measures = np.array(measures_scaled[test_index])
                        test_states = np.array(states[test_index])
                        
                        for measure_to_shuffle in shuffle_schedule[condition]:
                            train_measures[:,measure_to_shuffle] = train_measures[permutation,measure_to_shuffle]
                        
                        # fit SVMs
                        processes.append(fit_and_score_SVM.remote(train_measures,train_states,test_measures,test_states,kernel=kernel))
                        
        
                    results = [ray.get(p) for p in processes]
                    results = np.array(results)                    
                    SVM_models[m,state_iter,run,:,ii] = results[:,0]
                    SVM_acc[m,state_iter,run,:,ii] = results[:,1]
                    # mouse_SVMS[state_iter,run,:,ii] = results[:,0]
                    mouse_SVM_acc[state_iter,run,:,ii] = results[:,1]
                    
    
    # np.save(base_folder + 'decoder_data\\' + subject + '_'+ kernel+'_SVM_models_1000.npy',mouse_SVMS)
    np.save(base_folder + 'decoder_data\\' + subject + '_'+ kernel+'_SVM_accuracy_1000.npy',mouse_SVM_acc)
    