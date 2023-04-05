# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:49:15 2021 by Daniel Hulsey

Functions for formatting task performance data and fitting models with the SSM package 
"""

from uobrainflex.nwb import loadbehavior as load
import ssm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import scipy
from PIL import Image
import glob
import os 
from scipy.stats import mannwhitneyu #non parametric distribution test
from scipy.stats import levene   #levene tests for differences in the variance of data of non normal distributions
from scipy.stats import kruskal  #non paremetric analysis of variance

cols = ['#ff7f00', '#4daf4a', '#377eb8', '#7E37B8','m','r','c']

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]

def format_choice_behavior_hmm(trial_data, trial_labels, drop_no_response=False, drop_distractor_trials=True):
    """
    Parameters
    ----------
    trial_data : dataframe
        output of load_trials for a single session.
    drop_no_response : Boolean
        indicates whether to drop no response trials
    Returns
    -------
    inpts : TYPE
        DESCRIPTION.
    true_choice : TYPE
        DESCRIPTION.

    """
    ## get stim values
    # get outcome values from dicitonary
    miss_ind = trial_labels['outcome']['incorrect_reject']
    right_stim = trial_labels['target_port']['right']
    left_stim = trial_labels['target_port']['left']
    hit = trial_labels['outcome']['hit']
    error = trial_labels['outcome']['miss']
    distractor_ind = trial_labels['stimulus_type']['distractor']
    
    if drop_distractor_trials:
        #remove distractor tirals
        trials = trial_data.drop(index = trial_data.index[trial_data['stimulus_type'] == distractor_ind])
    else:
        trials = trial_data
        
    if drop_no_response:
        # remove no response trials
        trials = trials.drop(index = trial_data.index[trial_data['outcome']==miss_ind].tolist())
        # trial_inds = trials.index #get indices of kept trial data
    else:
        trials = trials
        # trial_inds = trials.index#get ALL indices of trial data
            
    if trials.iloc[0]['target_modality']==trial_labels['target_modality']['auditory']:
        stim = trials['aud_stim'].values
    elif trials.iloc[0]['target_modality']==trial_labels['target_modality']['visual']:
        stim = trials['vis_stim'].values
        
        
    # ## build stim values for inpt
    # # build arrays for visual and auditory stimulus identities
    # vis_dir = np.zeros(len(trials))    
    # vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['left'])[0]]=-1
    # vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['right'])[0]]=1
    # vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['no stim'])[0]]=np.nan
    
    # aud_dir = np.zeros(len(trials))
    # aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['left'])[0]]=-1
    # aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['right'])[0]]=1
    # aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['no stim'])[0]]=np.nan
  
    # # get array of auditory values
    # aud_val = trials['auditory_stim_difficulty'].values

    # # get visual gabor angle if available. if not available, stim values are hard coded as appropriate
    # vis_val = trials['visual_stim_difficulty'].values
    # vis_stim_id = np.zeros(len(trials))
    # if any(trials.columns.values==['visual_gabor_angle']):
    #     stim_val = trials['visual_gabor_angle'].values
    #     stim_val = stim_val[~np.isnan(stim_val)]
    #     stim_val = ((stim_val - 45)/45)
    #     stim_val = np.flip(np.unique(abs(stim_val)))
    #     if len(stim_val)>0:
    #         for this_diff in np.flip(np.unique(vis_val)):
    #             ind = np.where(vis_val==this_diff)[0]
    #             vis_stim_id[ind]=stim_val[int(this_diff)]
    # else:
    #     vis_stim_id[vis_val==1]=.8
    #     vis_stim_id[vis_val==2]=.6
    #     vis_stim_id[vis_val==0]=1
    
    # # this assumes target modality is constant, and will only output stim vals for the target modality
    # # getting both values will be necessary for models incorporating distractors
    # if trials.iloc[0]['target_modality']==trial_labels['target_modality']['auditory']:
    #     stim = aud_val*aud_dir
    # elif trials.iloc[0]['target_modality']==trial_labels['target_modality']['visual']:
    #     stim = vis_stim_id*vis_dir
    
    ## this only works on target trials!! This should be created during labview data collection in the future
    # create true_choice array. 
    true_choice = np.full([len(trials),1],np.int(2)) 
    # step through each response and update true_choice
    for ind in range(len(trials)):
        # if hit and left stim
        if all([trials['outcome'].iloc[ind] == hit, trials['target_port'].iloc[ind] == left_stim]):
            true_choice[ind] = int(0)
        # if error and right stim
        if all([trials['outcome'].iloc[ind] == error, trials['target_port'].iloc[ind] == right_stim]):
            true_choice[ind] = int(0)
        # if hit and right stim
        if all([trials['outcome'].iloc[ind] == hit, trials['target_port'].iloc[ind] == right_stim]):    
            true_choice[ind] = int(1)
        # if error and left stim
        if all([trials['outcome'].iloc[ind] == error, trials['target_port'].iloc[ind] == left_stim]):    
            true_choice[ind] = int(1)
        if trials['outcome'].iloc[ind] == miss_ind:#if the outcome is a miss, true choice = 2    
            true_choice[ind] = int(2)
            
    trials['choice']=true_choice
    
    # create list of choices with [stimulus information, bias constant]
    inpts = [ np.vstack((stim,np.ones(len(stim)))).T ]
    
    return inpts, true_choice, trials


def compile_choice_hmm_data(sessions, get_behavior_measures=False, verbose=True):
    """

    Parameters
    ----------
    sessions : list
        nwbfilepaths for sessions to compile.

    Returns
    -------
    inpts : TYPE
        .
    true_choice : TYPE
        DESCRIPTION.

    """
    
    #generate lists to append with each sessions' data
    inpts=list([])
    true_choices=list([])
    hmm_trials =list([])
    #loop through sessions to load and format data
    for sess in sessions:
        if verbose:
            print(f'Loading {sess}')
        nwbFileObj = load.load_nwb_file(str(sess))
        trial_data, trial_labels = load.read_trial_data(nwbFileObj)
        trial_data['file_name'] = os.path.basename(sess)
        these_inpts, these_true_choices, trials = format_choice_behavior_hmm(trial_data, trial_labels)
        if get_behavior_measures:
            behavior_measures = load.read_behavior_measures(nwbFileObj)
            _ ,behavior_events = load.read_behavior_events(nwbFileObj)
            trials = load.fetch_trial_beh_measures(trials, behavior_measures)
            trials = load.fetch_trial_beh_events(trials, behavior_events)
        # these_inpts, these_true_choices, trials = format_choice_behavior_hmm(trial_data, trial_labels)
        inpts.extend(these_inpts)
        true_choices.extend([these_true_choices])
        hmm_trials.extend([trials])
        
    return inpts, true_choices, hmm_trials


def choice_hmm_sate_fit(subject, inpts, true_choices, max_states=4, save_folder=''):
    # inpts, true_choices = compile_choice_hmm_data(sessions)
    ### State Selection
    #Create kfold cross-validation object which will split data for us
    num_sess = len(true_choices)
    from sklearn.model_selection import KFold
    nKfold = np.min([num_sess,5]) # if there are more than 4 sessions, group sessions together to create 4 folds
    kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)
    
    #Just for sanity's sake, let's check how it splits the data
    #So 5-fold cross-validation uses 80% of the data to train the model, and holds 20% for testing
    for ii, (train_index, test_index) in enumerate(kf.split(true_choices)):
        print(f"kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
    # max_states = 4 # largest number of states allowed in the model selection
    N_iters = 1000 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    TOL=10**-6 # tolerance parameter (see N_iters)
    
    # initialized training and test loglik for model selection, and BIC
    ll_training = np.zeros((max_states,nKfold))
    ll_heldout = np.zeros((max_states,nKfold))
    ll_training_map = np.zeros((max_states,nKfold))
    ll_heldout_map = np.zeros((max_states,nKfold))
    # BIC = np.zeros((max_states))
    
    # # Set the parameters of the GLM-HMM
    obs_dim = 1           # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(true_choices)))    # number of categories for output
    input_dim = inpts[0].shape[1]                                    # input dimensions
    
    
    #Outer loop over the parameter for which you're doing model selection for
    for iS, num_states in enumerate(range(1,max_states+1)):
        
        #Inner loop over kfolds
        for iK, (train_index, test_index) in enumerate(kf.split(true_choices)):
            nTrain = len(train_index); nTest = len(test_index)#*obs_dim
            
            # split choice data and inputs
            training_data = [true_choices[i] for i in train_index]
            test_data = [true_choices[i] for i in test_index]
            training_inpts=[inpts[i] for i in train_index]
            test_inpts=[inpts[i] for i in test_index]
            
        
            #Create HMM object to fit: MLE
            print(num_states)
            xval_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                               observation_kwargs=dict(C=num_categories), transitions="standard")
            #fit on training data
            hmm_lls = xval_glmhmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
            #Compute log-likelihood for each dataset
            ll_training[iS,iK] = xval_glmhmm.log_probability(training_data, inputs=training_inpts)/nTrain
            ll_heldout[iS,iK] = xval_glmhmm.log_probability(test_data, inputs=test_inpts)/nTest
            
            #Create HMM object to fit: MAP
            # Instantiate GLM-HMM and set prior hyperparameters
            prior_sigma = 2
            prior_alpha = 2
            map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                           observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                         transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
            #fit on training data
            hmm_lls = map_glmhmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
            #Compute log-likelihood for each dataset
            ll_training_map[iS,iK] = map_glmhmm.log_probability(training_data, inputs=training_inpts)/nTrain
            ll_heldout_map[iS,iK] = map_glmhmm.log_probability(test_data, inputs=test_inpts)/nTest
    
    #plot ll calculations
    global cols
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    for iS, num_states in enumerate(range(1,max_states+1)):
        plt.plot((iS+1)*np.ones(nKfold),ll_training[iS,:], color=cols[0], marker='o',lw=0)
        plt.plot((iS+1)*np.ones(nKfold),ll_heldout[iS,:], color=cols[1], marker='o',lw=0)
        plt.plot((iS+1)*np.ones(nKfold),ll_training_map[iS,:], color=cols[2], marker='o',lw=0)
        plt.plot((iS+1)*np.ones(nKfold),ll_heldout_map[iS,:], color=cols[3], marker='o',lw=0)
    
    plt.plot(range(1,max_states+1),np.mean(ll_training,1), label="training_MLE", color=cols[0])
    plt.plot(range(1,max_states+1),np.mean(ll_heldout,1), label="test_MLE", color=cols[1])
    plt.plot(range(1,max_states+1),np.mean(ll_training_map,1), label="training_MAP", color=cols[2])
    plt.plot(range(1,max_states+1),np.mean(ll_heldout_map,1), label="test_MAP", color=cols[3])
    # plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
    plt.legend(loc="lower right")
    plt.xlabel("states")
    plt.xlim(0, max_states+1)
    plt.ylabel("Log Probability")
    plt.title(subject + ' Model Fitting')
    if save_folder !='':
        plt.savefig(save_folder + subject +'_state_number_fitting.png')
        plt.close()
    
    # #drop lowest kfold value and determine highest LL state for use in model selected
    # ll_heldout_droplow = np.zeros([ll_heldout.shape[0],ll_heldout.shape[1]-1])
    # for idx,ll in enumerate(ll_heldout):
    #     ll=np.delete(ll,np.where(ll == min(ll))[0])
    #     ll_heldout_droplow[idx,:]=ll
    
    # ll_heldout_map_droplow = np.zeros([ll_heldout_map.shape[0],ll_heldout_map.shape[1]-1])
    # for idx,ll in enumerate(ll_heldout_map):
    #     ll=np.delete(ll,np.where(ll == min(ll))[0])
    #     ll_heldout_map_droplow[idx,:]=ll
    
    # avg_ll = np.mean(ll_heldout_map_droplow,axis=1)
    avg_ll = np.mean(ll_heldout_map,axis=1)
    num_states = np.where(avg_ll == avg_ll.max())[0][0]+1
    return num_states


def choice_hmm_fit(subject, num_states, inpts, true_choices):
    ## Fit GLM-HMM with MAP estimation:
    # Set the parameters of the GLM-HMM
    obs_dim = true_choices[0].shape[1]          # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(true_choices)))    # number of categories for output
    input_dim = inpts[0].shape[1]                                    # input dimensions

    # Instantiate GLM-HMM and set prior hyperparameters (MAP version)
    prior_sigma = 2
    prior_alpha = 2
    
    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                         observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                         transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    
    N_iters = 1000
    fit_map_ll = hmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters,
                                tolerance=10**-4)
    return hmm

def generate_n_states_hmms(max_states, inpts, true_choices):  
    subject=[]
    hmms=[]
    for num_states in np.arange(1,max_states+1):
        hmms.append(choice_hmm_fit(subject, num_states, inpts, true_choices))
    return hmms

def hmm_state_testing(max_states,train_inpts,train_choices,test_inpts,test_choices):
    subject=[]
    hmms=[]
    lls=[]

    for num_states in np.arange(1,max_states+1):
        hmms.append(choice_hmm_fit(subject, num_states, train_inpts, train_choices))
    for hmm in hmms:
        lls.append(hmm.log_probability(test_choices,test_inpts)/np.concatenate(test_inpts).shape[0]*np.log(2))
    return hmms, lls


def get_posterior_probs(hmm, hmm_trials, occ_thresh = .8):
    #pass true data to the model and get state occupancy probabilities
    true_choices = get_true_choices_from_hmm_trials(hmm_trials)
    inpts = get_inpts_from_hmm_trials(hmm_trials)
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
                   for data, inpt
                   in zip(true_choices, inpts)]
    
    #update hmm_trials with state probabilities
    for sess_ind in range(len(hmm_trials)):
        
        columns = hmm_trials[sess_ind].columns
        del_cols = []        
        for col in columns :
            if col[:6]=='state_':
                 del_cols.append(col)
            
        # hmm_trials[sess_ind].drop[del_cols[0]]
        # test = hmm_trials[sess_ind]
        hmm_trials[sess_ind].drop(del_cols, axis=1, inplace=True)
        
        hmm_trials[sess_ind]['inpt'] = inpts[sess_ind][:,0]
        hmm_trials[sess_ind]['choice'] = true_choices[sess_ind]
        session_posterior_prob = posterior_probs[sess_ind] 
        np.max(session_posterior_prob,axis=1)
        trial_max_prob = np.max(session_posterior_prob, axis = 1)
        trial_state = np.argmax(session_posterior_prob, axis = 1).astype(float)
        trial_state[np.where(trial_max_prob<occ_thresh)[0]] = np.nan
        hmm_trials[sess_ind]['hmm_state'] = trial_state
        for state in range(session_posterior_prob.shape[1]):
            hmm_trials[sess_ind]['state_' + str(state+1) +'_prob'] = session_posterior_prob[:,state]

    return posterior_probs, hmm_trials

def get_posterior_probs_from_hmm_trials(hmm_trials):   
    posterior_probs=[]
    #update hmm_trials with state probabilities
    for session_trials in hmm_trials:
        columns = session_trials.columns
        posterior_probabilities=[]
        for col in columns :
            if col[:6]=='state_':
                if len(posterior_probabilities)==0:
                    posterior_probabilities = session_trials[col].values
                else:
                    posterior_probabilities = np.vstack([posterior_probabilities, session_trials[col].values])
        posterior_probs.append(np.array(posterior_probabilities))    
    return posterior_probs

def get_inpts_from_hmm_trials(hmm_trials):
    inpts=list([])
    for session in hmm_trials:
        stim = session['inpt'].values
        these_inpts = [ np.vstack((stim,np.ones(len(stim)))).T ]
        inpts.extend(these_inpts)
    return inpts

def get_true_choices_from_hmm_trials(hmm_trials):
    true_choices=list([])
    for session in hmm_trials:
        these_choices = np.full([len(session),1],np.int(2)) 
        these_choices[:,0] = session['choice'].values
        true_choices.append(these_choices)
    return true_choices

def set_hmm_trials_choices(hmm_trials):
    for session in hmm_trials:
        outcomes = session['outcome'].values
        inpts = np.array(session['inpt'].values)
        
        inpts[inpts>0]=1
        inpts[inpts<0]=-1
        
        choices = np.full(len(inpts),2)
        for i in range(len(outcomes)):
            if all([outcomes[i]==0,inpts[i]==1]):
                choices[i]=1
            elif all([outcomes[i]==0,inpts[i]==-1]):
                choices[i]=0
            elif all([outcomes[i]==1,inpts[i]==-1]):
                choices[i]=1
            elif all([outcomes[i]==1,inpts[i]==1]):
                choices[i]=0
                
        session['choice']=choices
    return hmm_trials
            

def get_true_choices_from_outcomes_hmm_trials(hmm_trials):
    true_choices=list([])
    for session in hmm_trials:
        these_choices = np.full([len(session),1],np.int(2)) 
        outcomes = session['outcome'].values
        inpts = session['inpt'].values
        for i, outcome in enumerate(outcomes):
            if outcome==0:
                if inpts[i]>0:
                    these_choices[i]=1
                elif inpts[i]<0:
                    these_choices[i]=0
            elif outcome==1:
                if inpts[i]>0:
                    these_choices[i]=0
                elif inpts[i]<0:
                    these_choices[i]=1                  
        true_choices.append(these_choices)
    return true_choices

def get_psychos(hmm_trials):  
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
    
    psycho = np.full((int(np.unique(all_trials['hmm_state'].dropna()).max()+1),3,len(np.unique(all_trials['inpt']))),np.nan)
    for state in np.unique(all_trials['hmm_state'].dropna()):
        for ind, choice in enumerate(np.unique(all_trials['choice'])):
            for ind2, this_inpt in enumerate(np.unique(all_trials['inpt'])):
                state_trials=len(all_trials.query("hmm_state==@state and inpt==@this_inpt"))
                if state_trials>0:
                    psycho[int(state),ind,ind2]=len(all_trials.query("choice==@choice and hmm_state==@state and inpt==@this_inpt"))/state_trials
    
    xvals = np.unique(all_trials['inpt'])
    return psycho, xvals   

def get_trials_psychos(all_trials):
    l=[]
    r=[]
    m=[]
    for stim in np.unique(all_trials['inpt']):
        lft = len(all_trials.query('inpt==@stim and choice==0'))
        rt = len(all_trials.query('inpt==@stim and choice==1'))
        mis = len(all_trials.query('inpt==@stim and choice==2'))
        n_trials = len(all_trials.query('inpt==@stim'))
        l.append(lft/n_trials)
        r.append(rt/n_trials)
        m.append(mis/n_trials)
        
    all_trial_psycho = np.vstack([l,r,m])
    xvals = np.unique(all_trials['inpt'])
    return all_trial_psycho, xvals
    

def compare_psychos(psycho1,psycho2):
    all_similarity=[]
    
    missing_values = np.unique(np.where(psycho1!=psycho1)[2])
    missing_values2 = np.unique(np.where(psycho2!=psycho2)[2])
    missing_vals = np.unique(np.concatenate([missing_values,missing_values2]))
    
    psycho1 = np.delete(psycho1,missing_vals,axis=2)
    psycho2 = np.delete(psycho2,missing_vals,axis=2)
    
    for state in psycho2:
        similarity=[]
        for state2 in psycho1:
            similarity.append(np.sum(abs(state-state2)))
        all_similarity.append(similarity)
    return np.array(all_similarity)


def hmm_trials_to_all_trials(hmm_trials):
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        # if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
    return all_trials

def hmm_trials_to_p_state(hmm_trials):
    all_trials = hmm_trials_to_all_trials(hmm_trials)
    state_trials = all_trials.query('hmm_state==hmm_state and post_hoc_pupil_diameter==post_hoc_pupil_diameter')
    state_trials['hmm_state'].iloc[np.where(state_trials['hmm_state']>2)[0]]=2

    bin_size=.02
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = state_trials.query('post_hoc_pupil_diameter>=@p and post_hoc_pupil_diameter<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
    x = np.arange(0,1,bin_size)
    return p_state, x

def get_optimal_pupil(hmm_trials):
    p_state, x = hmm_trials_to_p_state(hmm_trials)
    
    
    y= p_state[:,0]
    
    x2=x[y==y]
    y=y[y==y]

    coeffs = np.polyfit(x2,y,2)
    poly = np.poly1d(coeffs)
    #poly optimal
    # xs = np.arange(0,1,.01)
    # plt.plot(xs,poly(xs),'r',linewidth=2)
    # plt.ylim([0,1])
    # plt.xlim([.4,1])
    
    poly_optimal_pupil = -coeffs[1]/(coeffs[0]*2)
    # plt.plot([poly_optimal_pupil,poly_optimal_pupil],[0,1],'r:')
    
    
    # median optimal
    temp = p_state[:,0]
    h=[]
    for ii, d in enumerate(temp):
        if d==d:
            for iii in range(round(d*100)):
                h.append(x[ii])
    med_opt_pupil = np.median(h)
    # plt.plot([med_opt_pupil,med_opt_pupil],[0,1],'k:')
    return med_opt_pupil, poly_optimal_pupil
        
    

def permute_hmm(hmm, hmm_trials):
    psycho,_ = get_psychos(hmm_trials)
    
    # perfect_psychos = np.zeros((3,psycho.shape[1],psycho.shape[2]))
    # perfect_psychos[0,0,:int(perfect_psychos.shape[2]/2)]=1
    # perfect_psychos[0,1,int(perfect_psychos.shape[2]/2):]=1
    # perfect_psychos[1,0,:]=1
    # perfect_psychos[2,1,:]=1
    # # perfect_psychos[3,2,:]=1
    
    
    perfect_psychos = np.zeros((2,psycho.shape[1],psycho.shape[2]))
    perfect_psychos[0,0,:int(perfect_psychos.shape[2]/2)]=1
    perfect_psychos[0,1,int(perfect_psychos.shape[2]/2):]=1
    perfect_psychos[1,2,:]=1

    
    # if np.isnan(psycho).any():
    #     del_ind = np.unique(np.where(np.isnan(psycho))[2])
    #     psycho = np.delete(psycho,del_ind,axis=2)
    similarity = compare_psychos(perfect_psychos,psycho)
    state_similarity = np.nanargmin(similarity,axis=0)
    
    if len(state_similarity) < psycho.shape[0]:
        extra_states = np.delete(np.arange(psycho.shape[0]), state_similarity)
        state_similarity = np.append(state_similarity,extra_states)
    
    if len(np.unique(state_similarity)) != len(state_similarity):
        best_psycho_ind = np.nanargmin(similarity[:,0])
        state_options = np.arange(similarity.shape[0])
        state_options = np.delete(state_options,best_psycho_ind)
        state_similarity = np.concatenate((np.full(1,best_psycho_ind),state_options))
        
    perm = state_similarity
    # if len(perm)<len(similarity):
    #     missing_state = list(set(np.array(range(similarity.shape[0])))- set(perm))
    #     perm = np.append(perm,missing_state)
        
    hmm.permute(perm)
    
    return hmm      

def plot_GLM_weights(subject, hmm,save_folder=''):
       ## plot results
    params = hmm.observations.params
    num_states = params.shape[0]

    if params.shape[1]>1:
        input_dim = params.shape[1]
        fig = plt.subplots(1, input_dim, sharey='all', figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
        name_categories=['L','R','miss'] 
    else:
        input_dim = 1
        plt.figure()

    global cols
    # Plot covariate GLM weights by HMM state

    for iC in range(input_dim):
        plt.subplot(1, input_dim, iC+1)
        for k in range(num_states):
            if input_dim>1:
                y = params[k][iC]
            else:
                y = params[k][0].T*-1
            plt.plot(range(params.shape[2]), y, marker='o',
                     color=cols[k], linestyle='-',
                     lw=1.5, label="state " + str(k+1))
        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.legend()
        if input_dim>1:
            plt.title('Weights for {}'.format(name_categories[iC]), fontsize = 15)
        else:
            plt.title(subject + ' GLM Weights', fontsize=15)
            
    if save_folder!='':
        plt.savefig(save_folder + subject +'_GLM_Weights.png')
        plt.close()


def plot_session_posterior_probs(subject, hmm,true_choices,inpts, save_folder=''):
    # plot trial state probability by session
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
                       for data, inpt
                       in zip(true_choices, inpts)]
    
    num_states =len(hmm.observations.params)
    global cols    
    
    for sess_id in range(0,len(true_choices)):
        plt.figure()
        #sess_id = 2 #session id; can choose any index between 0 and num_sess-1
        for indst in range(num_states):
            plt.plot(posterior_probs[sess_id][:, indst], label="State " + str(indst + 1), lw=2,
                     color=cols[indst])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize = 10)
        plt.xlabel("trial #", fontsize = 15)
        plt.ylabel("p(state)", fontsize = 15)
        plt.title('session ' + str(sess_id))
        if save_folder!='':
            plt.savefig(save_folder + subject + f" Pstate_session{sess_id}.png")
            plt.close()

def plot_transition_matrix(subject, hmm, save_folder=''):
    # plot state transition matrix
    plt.figure(figsize=[8,8])
    recovered_trans_mat = np.exp(hmm.transitions.log_Ps)
    num_states = recovered_trans_mat.shape[0]
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=4)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xlabel('state t+1', fontsize = 15)
    plt.ylabel('state t', fontsize = 15)
    plt.xticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
    plt.yticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title(subject + " transition matrix", fontsize = 20)
    if save_folder!='':
        plt.savefig(save_folder + subject +" transition_matrix.png")
        plt.close()


def plot_state_occupancy(subject, hmm_trials, save_folder=''):

    global cols
    state = np.empty(0)
    for trials in hmm_trials:
        state = np.append(state, trials['hmm_state'].values)
    _, state_occupancies = np.unique(state[~np.isnan(state)], return_counts=True)
    state_occupancies = state_occupancies/len(state)
    state_occupancies = np.append(state_occupancies,1-sum(state_occupancies))

    fig = plt.figure(figsize=(6, 7), dpi=80, facecolor='w', edgecolor='k')
    for z, occ in enumerate(state_occupancies):
        if z == len(state_occupancies)-1:
            plt.bar(z, occ, width = 0.8, color = 'k', hatch='/', alpha=0.1)
        else:
            plt.bar(z, occ, width = 0.8, color = cols[z])
    plt.ylim((0, 1))
    xs = np.arange(0,len(state_occupancies))
    plt.xticks(xs, np.append(xs[1:],'undecided'), fontsize = 10, rotation=20)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0','', '0.5','', '1'], fontsize=10)
    plt.xlabel('state', fontsize = 15)
    plt.ylabel('frac. occupancy', fontsize=15)
    plt.title(subject + ' State Occupancy', fontsize=15)
    if save_folder !='':
        plt.savefig(save_folder + subject + "_state_occupancy.png")    
        plt.close()
    
    plt.figure()
    session_state_occupancy = pd.DataFrame(columns=xs)
    lg=[]
    for idx, trials in enumerate(hmm_trials):
        for state in xs[:-1]:
            session_state_occupancy.loc[idx,state] = len(np.where(trials["hmm_state"] == state)[0])/len(trials)
    session_state_occupancy.iloc[:,-1] = 1-np.sum(session_state_occupancy.iloc[:,:-1],axis=1)
    for state in session_state_occupancy.columns[:-1]:
        plt.plot(session_state_occupancy[state],color=cols[state])
        lg.append('state '+str(state+1))
    plt.plot(session_state_occupancy.iloc[:,-1],'k:')
    lg.append('undecided')
    plt.title(subject +' state occupancy by session')
    plt.ylabel('p(state)')
    plt.xlabel('session')
    plt.legend(lg)

    if save_folder !='':
        plt.savefig(save_folder + subject + "_state_occupancy_by_session.png")
        plt.close()

def plot_state_posteriors_CDF(subject, posterior_probs, save_folder=''):
    global cols     
    posterior_probs_concat = np.concatenate(posterior_probs)
    # get state with maximum posterior probability at particular trial:
    state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
    
    fig = plt.figure(figsize=(6, 7), facecolor='w', edgecolor='k')
    for state in np.unique(state_max_posterior):
        state_confidence = posterior_probs_concat[np.where(state_max_posterior==state),state]
        y,x = np.histogram(state_confidence,bins=100)
        plt.plot(x[:len(y)],np.cumsum(y/sum(y)),color=cols[state],linewidth=3)
    y,x = np.histogram(np.max(posterior_probs_concat,axis=1),bins=100)
    plt.plot(x[:len(y)],np.cumsum(y/sum(y)),'k--')
    plt.plot([.8,.8],[0,1],'k:')
    plt.ylim([0,1])
    plt.xlim([.3,1])
    plt.xlabel('Posterior Probability Value', fontsize = 15)
    plt.ylabel('Cumulative Probability', fontsize = 15)
    plt.title(subject + '\nMax State Posterior Values', fontsize = 20)
    if save_folder !='':
        plt.savefig(save_folder +  subject + "_max_state_Posterior_value.png")   
        plt.close()
     
                
def plot_state_psychometrics(subject,hmm_trials,save_folder=''):
    psycho, stim_vals = get_psychos(hmm_trials)
    lines =['--',':','-']
    symbol = ['<','>','s']
    symb_plot=[]
    
    plt.figure(figsize=(10, 6))
    for idx in range(psycho.shape[0]):
        plt.subplot(2,round((psycho.shape[0]+1)/2),idx+1)
        for idx2, psych in enumerate(psycho[idx,:,:]):
            symb_plot.append(plt.plot(stim_vals,psych.T,symbol[idx2],color=cols[idx])[0])    
            plt.plot(stim_vals,psych.T,lines[idx2],color=cols[idx])
        if idx==0 or idx==3:
            plt.ylabel('Choice probability',fontsize=12)
        if  [3,4,5].count(idx)>0:
            plt.xlabel('Stimulus weight',fontsize=12)
        if idx==0:
            plt.legend(symb_plot,['left','right','miss'])
    plt.suptitle(subject + ' psychometrics by state',fontsize=20)

    if save_folder !='':
        plt.savefig(save_folder + subject + "_choice_by_state.png")        
        plt.close()
    
        
def get_dwell_times(hmm_trials):
    session_transitions=[]
    all_transitions=pd.DataFrame()
    for sess, trials in enumerate(hmm_trials):
        transitions = pd.DataFrame(columns=['state','dwell','first_trial','last_trial'])
        trial_state = trials['hmm_state'].values
        state_ind = 0
        n=0
        state_start = n
        state=[]
        dwell=[]
        start_trial=[]
        end_trial=[]
        while n<len(trial_state)-1:
            this_state = trial_state[n]
            next_state = trial_state[n+1]
            if not(this_state == next_state):
                state_end = n+1
                if ~np.isnan(this_state):
                    state_ind = state_ind+1
                    dwell.append(state_end-state_start+1)
                    state.append(this_state)
                    start_trial.append(state_start)
                    end_trial.append(state_end)
                state_start = n+1
            n=n+1
        state_end = n
        if ~np.isnan(this_state):
            dwell.append(state_end-state_start)
            state.append(this_state)
            start_trial.append(state_start)
            end_trial.append(state_end)
        transitions['state']=state
        transitions['dwell']=dwell
        transitions['first_trial']=start_trial
        transitions['last_trial']=end_trial
        session_transitions.append(transitions)
        all_transitions = all_transitions.append(transitions)
    return session_transitions
      

def plot_dwell_times(subject, hmm_trials, save_folder = ''):
    global cols
    session_transitions=[]
    all_transitions=pd.DataFrame()
    for sess, trials in enumerate(hmm_trials):
        transitions = pd.DataFrame(columns=['state','dwell','first_trial','last_trial'])
        trial_state = trials['hmm_state'].values
        state_ind = 0
        n=0
        state_start = n
        state=[]
        dwell=[]
        start_trial=[]
        end_trial=[]
        while n<len(trial_state)-1:
            this_state = trial_state[n]
            next_state = trial_state[n+1]
            if not(this_state == next_state):
                state_end = n+1
                if ~np.isnan(this_state):
                    state_ind = state_ind+1
                    dwell.append(state_end-state_start+1)
                    state.append(this_state)
                    start_trial.append(state_start)
                    end_trial.append(state_end)
                state_start = n+1
            n=n+1
        state_end = n
        if ~np.isnan(this_state):
            dwell.append(state_end-state_start)
            state.append(this_state)
            start_trial.append(state_start)
            end_trial.append(state_end)
        transitions['state']=state
        transitions['dwell']=dwell
        transitions['first_trial']=start_trial
        transitions['last_trial']=end_trial
        session_transitions.append(transitions)
        all_transitions = all_transitions.append(transitions)
        
    plt.figure()
    lgnd =[]
    for this_state in np.unique(all_transitions['state']):
        dwell_index = np.where(all_transitions['state']==this_state)[0]
        these_dwells = all_transitions['dwell'].iloc[dwell_index].values
        dwell_hist, bins = np.histogram(these_dwells,range(0,200,10))
        plt.plot(bins[:-1],dwell_hist/sum(dwell_hist),cols[int(this_state)])
        lgnd.append('State ' + str(int(this_state+1)))
    plt.xlabel('State Dwell Time \n # trials')
    plt.ylabel('Fraction in State')
    plt.title(subject + ' State Dwell Times')
    plt.legend(lgnd)
    
    if save_folder == 0:
        plt.close()
    elif save_folder != '':
        plt.savefig(save_folder +  subject + "_state_dwelltime_distribution.png")       
        plt.close()

        
    return session_transitions



def plot_session_summaries(subject, hmm_trials, nwbfilepaths,save_folder ='', measure_hist = True):
    global cols
    for session, these_trials in enumerate(hmm_trials):
        columns = these_trials.columns
        # plt.figure()    
        fig = plt.figure(figsize=(25, 12), facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        gs = gridspec.GridSpec(1,4)
        if measure_hist:
            ax.set_position(gs[0:3].get_position(fig))        
        plt.plot(these_trials['state_1_prob'].values,cols[0],linewidth=3)
        ax.plot(these_trials['state_2_prob'].values,cols[1],linewidth=3)
        lg = ['state 1','state 2']
        if any(columns=='state_3_prob'):
            ax.plot(these_trials['state_3_prob'].values,cols[2],linewidth=3)
            lg.append('state 3')
        if any(columns=='state_4_prob'):
            ax.plot(these_trials['state_4_prob'].values,cols[3],linewidth=3)
            lg.append('state 4')
        if any(columns=='state_5_prob'):
            ax.plot(these_trials['state_5_prob'].values,cols[4],linewidth=3)
            lg.append('state 5')
        if any(columns=='state_6_prob'):
            ax.plot(these_trials['state_6_prob'].values,cols[5],linewidth=3)
            lg.append('state 6')
        if 'post_hoc_pupil_diameter' in columns:
            pupil = these_trials['post_hoc_pupil_diameter']
            pupil = pupil/2+.5
            ax.scatter(range(len(these_trials)),pupil,color='k')
            lg.append('pupil')
        elif 'pupil_diameter' in columns:
            pupil = these_trials['pupil_diameter']
            pupil = pupil/2+.5
            ax.scatter(range(len(these_trials)),pupil,color='k')
            lg.append('pupil')
        if 'running_speed' in columns:
            run = these_trials['running_speed']
            run = run/max(run)/6+.33
            ax.scatter(range(len(these_trials)),run,color='k',marker = ">")
            lg.append('wheel')

        if 'whisker_energy' in columns:
            whisk = these_trials['whisker_energy']
            whisk = whisk/max(whisk)/6+.16
            ax.scatter(range(len(these_trials)),whisk,color='k',marker = "s")
            lg.append('whisk')

        if 'reaction_time' in columns:
            rt = these_trials['reaction_time']
            rt = rt/np.nanmax(rt)/6
            ax.scatter(range(len(these_trials)),rt,color='k',marker = "*")
            lg.append('reaction time')
    
        ax.legend(lg)
        plt.xlabel('Trial',fontsize = 30)
        plt.ylabel('p(state)')
        plt.suptitle(these_trials['stage'].iloc[0] + '    ' + nwbfilepaths[session],fontsize=17)
        
        if measure_hist:
            gs = gridspec.GridSpec(8,4)
            glist = [7,15,23,31]
            hist_lim = [1,.5,1,2500]
            hist_bins = [.025,.01,.025,50]
            xlabels = ['pupil diameter (% max)','running speed (m/s)','whisker energy (a.u.)','reaction time (ms)']
            for i,measure in enumerate(['post_hoc_pupil_diameter', 'running_speed', 'whisker_energy', 'reaction_time']):
                if measure in columns:
                    ax = fig.add_subplot(gs[glist[i]])
                    for state in np.unique(these_trials['hmm_state'].dropna()):
                        measure_data = these_trials.query('hmm_state == @state')[measure]
                        hist_y,hist_x = np.histogram(measure_data,bins = np.arange(0,hist_lim[i],hist_bins[i]))
                        ax.plot(hist_x[1:],hist_y/sum(hist_y),color=cols[int(state)])
                        # plt.title(measure)
                        plt.xlabel(xlabels[i])
                    plt.ylabel('fraction trials')
            
        if save_folder != '':
            plt.savefig(save_folder +  subject + "_session_" + str(session) + "_summary.png")   
            plt.close()


def plot_session_summaries_patch(subject, hmm_trials,dwell_times,nwbfilepaths,save_folder =''):
    global cols
    for session, these_trials in enumerate(hmm_trials):
        columns = these_trials.columns
        hand=[]
        lg=[]
        dwells = dwell_times[session]
        fig = plt.figure(figsize=(25, 12), facecolor='w', edgecolor='k')
        ax = fig.add_axes([0.1, .1, 0.8, .8])

        if len(dwells.index)>0:
            for idx in dwells.index: 
                ax.add_patch(Rectangle((dwells.loc[idx,'first_trial']-.4,0), dwells.loc[idx,'dwell']-1.1, 1,color=cols[int(dwells.loc[idx,'state'])],alpha=.2))
            for idx in range(0,int(max(dwells['state']))+1):
                hand.append(ax.add_patch(Rectangle((0,0),0,0,color=cols[idx],alpha=.2)))
                lg.append('state ' + str(idx+1))
        if 'post_hoc_pupil_diameter' in columns:
            pupil = these_trials['post_hoc_pupil_diameter']
            pupil = pupil/2+.5
            hand.append(plt.scatter(range(len(these_trials)),pupil,color='k'))
            lg.append('pupil')
        if 'running_speed' in columns:
            run = these_trials['running_speed']
            run = run/max(run)/6+.33
            hand.append(plt.scatter(range(len(these_trials)),run,color='k',marker = ">"))
            lg.append('wheel')
        if 'whisker_energy' in columns:
            whisk = these_trials['whisker_energy']
            whisk = whisk/max(whisk)/6+.16
            hand.append(plt.scatter(range(len(these_trials)),whisk,color='k',marker = "s"))
            lg.append('whisk')
        if 'reaction_time' in columns:
            rt = these_trials['reaction_time']
            rt = rt/np.nanmax(rt)/6
            hand.append(plt.scatter(range(len(these_trials)),rt,color='k',marker = "*"))
            lg.append('reaction time')

        plt.legend(hand,lg)
        plt.xlabel('Trial',fontsize = 30)
        plt.title(nwbfilepaths[session],fontsize=17)
        if save_folder != '':
            plt.savefig(save_folder +  subject + "_session_" + str(session) + "_summary.png")         
            plt.close()
         
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    a = a[a==a]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    ci = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, ci   
         
def plot_locomotion_by_state(subject, hmm_trials, save_folder = ''):
    cutoff = .01
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
        
    #by session
    locomotion_data = pd.DataFrame(columns = np.unique(all_trials['hmm_state'].dropna()))
    n_session=[]
    for i, trials in enumerate(hmm_trials):
        for state in np.unique(trials['hmm_state'].dropna()):
            run_vals = trials.query('hmm_state == @state')['running_speed']
            locomotion_data.loc[i,state] = len(np.where(run_vals>cutoff)[0])/len(run_vals)
       

    
    plt.figure()
    ax = plt.subplot()
    x_lb=[]
    for state in locomotion_data.columns:
        this_data = [locomotion_data[state].dropna().values.astype(np.float)]
        n_session=len(this_data[0])
        x_lb.append('state ' + str(int(state+1)) + '\nn = ' + str(n_session))
        parts = ax.violinplot(this_data,[int(state)],showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(cols[int(state)])
            pc.set_edgecolor('black')
            pc.set_alpha(.7)
    
            
            
    # plt.figure()
    for state in locomotion_data.columns:
        this_data = locomotion_data[state].dropna()
        # plt.scatter(state + np.random.rand(len(this_data))/2-.25,this_data, color = cols[int(state)], alpha = .4,s=10)
        plt.scatter(state + np.random.rand(len(this_data))/2-.25,this_data, color = 'k',s=10,alpha=.7)
        plt.scatter(state,np.mean(this_data),color = 'k', marker = 's',s=50)
        plt.errorbar(state,np.mean(this_data),scipy.stats.sem(this_data),color = 'k',linewidth=3)
        # plt.errorbar(state,np.mean(this_data),np.std(this_data),color = 'k',linewidth=1)
    
    plt.xticks(ticks = locomotion_data.columns,labels = x_lb)
    plt.ylabel('proportion trials locomoting')
    plt.title(subject + ' locomotion probabiltiy by session')
    plt.ylim([0,1])
    if save_folder != '':
        plt.savefig(save_folder +  subject + "_running_trials_violin.png")         
        plt.close()
        
    # plt.figure()
    # for state in locomotion_data.columns:
    #     this_data = locomotion_data[state].dropna()
    #     plt.scatter(state + np.random.rand(len(this_data))/4-.25,this_data, color = cols[int(state)], alpha = .4,s=15)
    #     # plt.scatter(state + np.random.rand(len(this_data))/2-.25,this_data, color = 'k',s=10)
    #     plt.scatter(state,np.mean(this_data),color = 'k', marker = 's')
    #     plt.errorbar(state,np.mean(this_data),scipy.stats.sem(this_data),color = 'k',linewidth=3)
    #     plt.errorbar(state,np.mean(this_data),np.std(this_data),color = 'k',linewidth=1)            
    
    # if save_folder != '':
    #     plt.savefig(save_folder +  subject + "_running_trials_scatter.png")         
    #     plt.close()

def plot_whisk_by_state(subject, hmm_trials, save_folder = ''):
    cutoff = .15
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
        
    #by session
    whisk_data = pd.DataFrame(columns = np.unique(all_trials['hmm_state'].dropna()))
    n_session=[]
    for i, trials in enumerate(hmm_trials):
        for state in np.unique(trials['hmm_state'].dropna()):
            whisk_vals = trials.query('hmm_state == @state')['whisker_energy']
            whisk_data.loc[i,state] = len(np.where(whisk_vals>cutoff)[0])/len(whisk_vals)
       

    
    plt.figure()
    ax = plt.subplot()
    x_lb=[]
    for state in whisk_data.columns:
        this_data = [whisk_data[state].dropna().values.astype(np.float)]
        n_session=len(this_data[0])
        x_lb.append('state ' + str(int(state+1)) + '\nn = ' + str(n_session))
        parts = ax.violinplot(this_data,[int(state)],showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(cols[int(state)])
            pc.set_edgecolor('black')
            pc.set_alpha(.7)
    
            
            
    # plt.figure()
    for state in whisk_data.columns:
        this_data = whisk_data[state].dropna()
        # plt.scatter(state + np.random.rand(len(this_data))/2-.25,this_data, color = cols[int(state)], alpha = .4,s=10)
        plt.scatter(state + np.random.rand(len(this_data))/2-.25,this_data, color = 'k',s=10,alpha=.7)
        plt.scatter(state,np.mean(this_data),color = 'k', marker = 's',s=50)
        plt.errorbar(state,np.mean(this_data),scipy.stats.sem(this_data),color = 'k',linewidth=3)
        # plt.errorbar(state,np.mean(this_data),np.std(this_data),color = 'k',linewidth=1)
    
    plt.xticks(ticks = whisk_data.columns,labels = x_lb)
    plt.ylabel('proportion trials whisking')
    plt.title(subject + ' whisk probabiltiy by session')
    plt.ylim([0,1])
    if save_folder != '':
        plt.savefig(save_folder +  subject + "_whisk_trials_violin.png")         
        plt.close()

def plot_measures_by_state_v2(subject, hmm_trials, save_folder = ''):
        
    fsize = 20
    measures = pd.DataFrame(columns=['pupil_diameter','running_speed','whisker_energy','reaction_time','pupil_diff','hmm_state','session'])
    for idx, trials in enumerate(hmm_trials):
        trials['session']=idx
        measures = measures.append(trials[measures.columns])
    
    bin_size = {'pupil_diameter' : .05,
                'whisker_energy' : .02,
                'running_speed' : .02,
                'reaction_time' : 50,
                'pupil_diff' : .01,}
    xlabel = {'pupil_diameter' : 'pupil diameter (%max)',
                'whisker_energy' : 'whisker energy (a.u.)',
                'running_speed' : 'wheel speed (m/s)',
                'reaction_time' : 'reaction time (ms)',
                'pupil_diff' : 'trial-to-trial pupil difference (%max)'}
    for measure in measures.columns[:-2]:
        plt.figure(figsize=(25, 12), facecolor='w', edgecolor='k')
        ax = plt.subplot(1,3,1)
        all_data=[]
        lg=[]
        #plot histograms of measures by state
        for i, state in enumerate(np.unique(measures['hmm_state'].dropna())):
            data = measures.query('hmm_state == @state')[measure].dropna()
            bins = np.arange(measures[measure].min(),measures[measure].max(),bin_size[measure])
            data_h, _ = np.histogram(data,bins)
            plt.plot(bins[:-1],data_h/sum(data_h),linewidth=3,color = cols[i])
            all_data.append(data.values)
            lg.append('state ' + str(int(state+1)))
        plt.legend(lg,fontsize = 20)
        plt.xlabel(xlabel[measure],fontsize = fsize)
        plt.ylabel('probability',fontsize = fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        
        
        alpha = .05/(i+1)
        
        mw = np.full([len(all_data),len(all_data)],np.nan)
        lv = np.full([len(all_data),len(all_data)],np.nan)
        mw = np.full([len(all_data),len(all_data)],alpha)
        lv = np.full([len(all_data),len(all_data)],alpha)
        for i in range(len(all_data)):
            for k in range(i+1,len(all_data)):
                s,mw[i,k]=mannwhitneyu(all_data[i],all_data[k])
                s, lv[i,k] = levene(all_data[i], all_data[k])   
                
        mw[mw>alpha]=alpha
        lv[lv>alpha]=alpha
        mw[-1,0]=0
        lv[-1,0]=0
    
        #plot mannwhitney test results
        ax = plt.subplot(1,3,2)
        im = ax.imshow(mw,cmap='inferno')
        plt.yticks(range(0,k+1),range(1,k+2))
        plt.xticks(range(0,k+1),range(1,k+2))
        plt.title('Mann–Whitney U',fontsize=fsize)
        plt.ylabel('hmm state',fontsize=fsize)
        plt.xlabel('hmm state',fontsize=fsize)
        plt.colorbar(im)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        
        #plot levene test results
        ax = plt.subplot(1,3,3)
        im = ax.imshow(lv,cmap='inferno')
        plt.yticks(range(0,k+1),range(1,k+2))
        plt.xticks(range(0,k+1),range(1,k+2))
        plt.title('levene',fontsize=fsize)
        plt.ylabel('hmm state',fontsize=fsize)
        plt.xlabel('hmm state',fontsize=fsize)
        plt.colorbar(im)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        
        s, p = kruskal(all_data[0],all_data[1],all_data[2])
        plt.suptitle(measure + '\n\nkruskal wallace p value = ' + str(p),fontsize=fsize)
        if save_folder !='':
            plt.savefig(save_folder + subject +'_' + measure +"_stats_v2.png")
            plt.close()
    
def plot_measures_by_state(subject, hmm_trials, save_folder = ''):
    global cols
    xlabels = {'pupil_diameter': 'Pupil Diameter (% max)',
               'running_speed': 'Wheel Speed (m/s)',
               'whisker_energy': 'Whisker Motion Energy (a.u)',
               'reaction_time': 'Reaction Time (ms)',
               'pupil_diff': 'Trial Delta Pupil (%max)',
               'hmm_state': 'HMM state'}
    titles = {'pupil_diameter': 'Pupil',
             'running_speed': 'Wheel speed',
             'whisker_energy': 'Whisker movement',
             'reaction_time':'Reaction time',
             'pupil_diff': 'Pupil change from last trial',
             'hmm_state':''}
    
    measures = pd.DataFrame(columns=['pupil_diameter','running_speed','whisker_energy','reaction_time','pupil_diff','hmm_state'])
    for trials in hmm_trials:
        concat_measures = np.append(trials.columns.values, measures.columns.values)
        value, count = np.unique(concat_measures,return_counts = True)
        these_measures = value[count>1]
        measures = measures.append(trials[these_measures])
    states = np.unique(measures['hmm_state'].dropna())
    legends =[]
    for state in states:
        n= len(measures.query('hmm_state==@state'))
        legends.append('State ' + str(int(state+1)) + ' (n = ' + str(int(n))+')')
 
    for measure in measures.columns:
        plt.figure()
        this_data = measures[measure].dropna()
        if measure== 'pupil_diff':
            this_data = abs(this_data)
        
        if measure == 'running_speed':
            cutoff = .01
        elif measure == 'whisker_energy':
            cutoff = .15
        else:
            cutoff = np.nan
                    
        bins = np.arange(min(this_data),max(this_data),np.std(this_data)/5)
        for state in states:
            data_for_hist = measures.query('hmm_state == @state')[measure].values
            # if ~np.isnan(cutoff):
            #     active = len(np.where(data_for_hist>cutoff)[0])/len(data_for_hist)
            #     passive = 1-active
            #     data_for_hist = data_for_hist[np.where(data_for_hist>cutoff)[0]]  
            hist_data = np.histogram(data_for_hist,bins=bins)
            plt.plot(hist_data[1][:-1],hist_data[0]/sum(hist_data[0]),color=cols[int(state)],linewidth=3)
            # plt.plot(hist_data[1][:-1],1-np.cumsum(hist_data[0])/sum(hist_data[0]),color=cols[int(state)],linewidth=3)
        plt.title(measure)
        plt.title(subject + '\n' + titles[measure] + ' distribution by state',fontsize=15)
        plt.xlabel(xlabels[measure],fontsize=12)
        plt.ylabel('Probability',fontsize=12)
        plt.legend(legends)
        if save_folder !='':
            plt.savefig(save_folder + subject +'_' + measure +"_by_state.png")
            plt.close()
        
        plt.figure()
        for state in states: 
            if measure == 'running_speed':
                cutoff = .01
            elif measure == 'whisker_energy':
                cutoff = .15
            else:
                cutoff = np.nan
                    
            data_for_stat = measures.query('hmm_state == @state')[measure].dropna().values

            if ~np.isnan(cutoff):
                active = len(np.where(data_for_stat>cutoff)[0])/len(data_for_stat)
                passive = 1-active
                data_for_stat = data_for_stat[np.where(data_for_stat>cutoff)[0]]  
            if measure== 'pupil_diff':
                data_for_stat = abs(data_for_stat)        
            
            m, ci = np.mean(data_for_stat),np.std(data_for_stat)
            plt.plot(state,m,'s',color=cols[int(state)])
            plt.errorbar(state,m,ci,color=cols[int(state)])
        plt.xticks(range(0,len(states)),range(1,len(states)+1))
        plt.xlabel('HMM State')
        plt.ylabel(xlabels[measure])
        bot, top= plt.ylim()
        if bot>0:
            bot = bot/1.1
        plt.ylim([bot,top])
        plt.legend(legends)
        plt.title(subject + '\n' + titles[measure] + ' statistics by state',fontsize=15)
        if save_folder !='':
            plt.savefig(save_folder + subject +'_' + measure +"_stats.png")
            plt.close()
            
         
def plot_measures_space_state(subject, hmm_trials, save_folder = ''):
    global cols
    measures = pd.DataFrame(columns=['pupil_diameter','running_speed','whisker_energy','reaction_time','pupil_diff','hmm_state','session'])
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#7E37B8','m','r','c']
    
    for idx, trials in enumerate(hmm_trials):
        trials['session']=idx
        measures = measures.append(trials[measures.columns])
    
    
    plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    for i, measure in enumerate(measures.columns[:-2]):
        for j in range(i,len(measures.columns[:-2])):
            n= i * len(measures.columns[:-2]) + j +1
            plt.subplot(5,5,n)
            measure2 = measures.columns[j]
            plt.title(measure + ' x ' + measure2)
                
            for state in np.unique(measures['hmm_state'].dropna()):
                state_data = measures.query('hmm_state == @state')
                data1 = state_data[measure].values
                data2 = state_data[measure2].values
                
                plt.scatter(data1,data2,s=4,color=cols[int(state)])
    plt.suptitle('measures space by state', fontsize = 20)
    if save_folder !='':
        plt.savefig(save_folder + subject +'_measures_space_by_state.png')
        plt.close()
          
def plot_hmm_state_by_measure_quantiles(subject, hmm_trials, save_folder =''):
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
    
    measures = ['pupil_diameter','whisker_energy','running_speed','pupil_diff']

    for measure in measures:
        stim_data = all_trials[measure].dropna().values
        
        hmm_state = np.full([6,20],np.nan)
        hmm_state_percent = np.full([6,20],np.nan)
        for i,qt in enumerate(np.arange(0,1,.1)):
            lower = np.quantile(stim_data,qt)
            upper = np.quantile(stim_data,qt+.1)
            
            these_trials = all_trials.query(measure + '> @lower and ' + measure + ' < @upper ')
            # performance.get_trial_psychos(these_trials)
            # plt.figure()
            # performance.plot_trial_psycho(these_trials)
            
            for j in np.unique(these_trials['hmm_state'].dropna()):
                hmm_state[int(j),i]=len(these_trials.query('hmm_state == @j'))
                hmm_state_percent[int(j),i]=len(these_trials.query('hmm_state == @j'))/len(these_trials)
        
        plt.figure()
        for i, state_data in enumerate(hmm_state_percent):
            plt.plot(state_data,color = cols [i])
        plt.ylabel('p(state)')
        plt.xticks(range(0,10),np.arange(0,100,10))
        plt.xlabel(measure + ' quantile')
        if save_folder !='':
            plt.savefig(save_folder + subject + '_state_probability_by_' + measure + '_quantile.png')
            plt.close()

            
def state_measures_by_session_95ci(subject, hmm_trials, save_folder = ''):
    measures = pd.DataFrame(columns=['pupil_diameter','running_speed','whisker_energy','reaction_time','hmm_state','session'])
    
    for idx, trials in enumerate(hmm_trials):
        trials['session']=idx
        measures = measures.append(trials[measures.columns])
                
    states = np.unique(measures['hmm_state'].dropna())
    plt.figure()
    
    for measure in list(measures.columns[:-2]):
        all_stats = pd.DataFrame(columns=['mean','error','state'])
        stats = pd.DataFrame(columns=['mean','error','state'])
        for state in states:
            for session in np.unique(measures['session']):
                    # measure = measures.columns[1]
                    this_data = measures.query('hmm_state == @state and session == @session')[measure].values
                    m, ci = mean_confidence_interval(this_data, confidence=0.95)
                    # plt.errorbar(session,m,ci,color=cols[int(state)])
                    # plt.scatter(session,m,color=cols[int(state)])
                    stats.loc[session,'mean']=m
                    stats.loc[session,'error']=ci
                    stats.loc[session,'state']=state
            all_stats = all_stats.append(stats)
            
        
        plt.figure()
        for state in states:        
            this_data = all_stats.query('state==@state')
            m, ci = mean_confidence_interval(this_data['mean'].values, confidence=0.95)
            plt.errorbar(state,m,ci,color=cols[int(state)])#,color=cols[int(state)])
            plt.scatter(state,m,color=cols[int(state)])
            plt.title(measure + ' session means')
            
        plt.figure()
        for state in states:        
            this_data = all_stats.query('state==@state')
            m, ci = mean_confidence_interval(this_data['error'].values, confidence=0.95)
            plt.scatter(np.ones(len(this_data))*state,this_data['error'].values,color=cols[int(state)],s=10)
            plt.errorbar(state,m,ci,color='k')#cols[int(state)])#,color=cols[int(state)])
            plt.scatter(state,m,color=cols[int(state)],s=100)
        
            plt.title(measure + ' session variance')

                      
def state_measures_by_session_std(subject, hmm_trials, save_folder = ''):
    measures = pd.DataFrame(columns=['pupil_diameter','running_speed','whisker_energy','reaction_time','hmm_state','session'])
    
    for idx, trials in enumerate(hmm_trials):
        trials['session']=idx
        measures = measures.append(trials[measures.columns])
                
    states = np.unique(measures['hmm_state'].dropna())
    plt.figure()
    
    for measure in list(measures.columns[:-2]):
        all_stats = pd.DataFrame(columns=['mean','error','state'])
        stats = pd.DataFrame(columns=['mean','error','state'])
        for state in states:
            for session in np.unique(measures['session']):
                    # measure = measures.columns[1]
                    this_data = measures.query('hmm_state == @state and session == @session')[measure].values
                    m = np.nanmean(this_data)
                    std = np.nanstd(this_data)
                    # plt.errorbar(session,m,ci,color=cols[int(state)])
                    # plt.scatter(session,m,color=cols[int(state)])
                    stats.loc[session,'mean']=m
                    stats.loc[session,'error']=std
                    stats.loc[session,'state']=state
            all_stats = all_stats.append(stats)
            
        
        fig = plt.figure()
        ax = fig.add_axes([.1,.1,.8,.8])
        for state in states:        
            this_data = all_stats.query('state==@state')
            m = np.nanmean(this_data['mean'].values)
            std = np.nanstd(this_data['mean'].values)
            
            data = [this_data['mean'].dropna().values.astype(np.float)]
            parts = ax.violinplot(data,[int(state)],showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(cols[int(state)])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
            # ax.scatter(np.ones(len(this_data))*state,this_data['mean'].values,color=cols[int(state)],s=10)
            # ax.errorbar(state,m,std,color='k')#cols[int(state)])#,color=cols[int(state)])
            # ax.scatter(state,m,color=cols[int(state)],s=100)
            plt.title(measure + ' session means')
            
        plt.figure()
        for state in states:        
            this_data = all_stats.query('state==@state')
            m = np.nanmean(this_data['error'].values)
            std = np.nanstd(this_data['error'].values)
            plt.scatter(np.ones(len(this_data))*state,this_data['error'].values,color=cols[int(state)],s=10)
            plt.errorbar(state,m,std,color='k')#cols[int(state)])#,color=cols[int(state)])
            plt.scatter(state,m,color=cols[int(state)],s=100)
        
            plt.title(measure + ' session variance')
            
            
def p_optimal_by_feature(subject , hmm_trials, save_folder = ''):
    
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        if np.unique(trials['hmm_state'])[0]==0:
            print(1)
            all_trials = all_trials.append(trials)

    pupil_measure = 'pupil_diameter'    
    if np.isin('post_hoc_pupil_diameter',all_trials.columns):
        pupil_measure = 'post_hoc_pupil_diameter'
    
    pupil_probs = np.full([3,50],np.nan)
    face_probs = np.full([3,50],np.nan)
    whisk_probs = np.full([3,50],np.nan)
    run_probs = np.full([3,50],np.nan)
        
    state_trials = all_trials.query('hmm_state==hmm_state')
    
    
    # good_pupil = state_trials.query('post_hoc_pupil_diameter==post_hoc_pupil_diameter')
    good_pupil = state_trials.query(pupil_measure + '==' + pupil_measure)
    good_pupil['hmm_state'].iloc[np.where(good_pupil['hmm_state']>2)[0]]=2
    bin_size=.02
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = good_pupil.query(pupil_measure + '>=@p and ' + pupil_measure + '<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    pupil_probs[int(state[j]),k]=counts[int(j)]/sum(counts)
    
    
    bin_size=.05
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = good_pupil.query('whisker_energy>=@p and whisker_energy<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    whisk_probs[int(state[j]),k]=counts[int(j)]/sum(counts)
    
    bin_size=.05
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = good_pupil.query('face_energy>=@p and face_energy<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    face_probs[int(state[j]),k]=counts[int(j)]/sum(counts)
    
    
    bin_size=.02
    p_state=np.full([50,3],np.nan)
    for k,p in enumerate(np.arange(-.02,.3,bin_size)):
        p2 = p+bin_size
        if p==.28:
            p2 = 1
        if p==-.02:
            p=-1
        p_trials = good_pupil.query('running_speed>=@p and running_speed<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    run_probs[int(state[j]),k]=counts[int(j)]/sum(counts)
    
              
    feat = ['pupil','face','whisk','run']                 
    plt.figure()
    for feature,data in enumerate([pupil_probs,face_probs,whisk_probs,run_probs]):
         plt.subplot(2,4,feature+1)
         for i in range(3):
             plt.plot(data[i,:],color=cols[i])
         plt.title(feat[feature])
         plt.xticks(range(50),np.arange(0,1,bin_size))
         
    feat = [pupil_measure,'face_energy','whisker_energy','running_speed'] 
    for i, feature in enumerate(feat):
        data = all_trials[feature]
        plt.subplot(2,4,i+5)
        plt.hist(data)
    
    plt.suptitle(subject,fontsize=15)
    
    if save_folder !='':
        plt.savefig(save_folder + subject + "_p_opt.png")
        plt.close()    

def report_cover(subject, nwbfilepath,save_folder):
    trial_data, trial_dict = load.load_trial_data(nwbfilepath)
    
    if trial_dict['target_modality']['visual']==int(trial_data['target_modality'][0]):
        modality = 'Visual Task'
    elif trial_dict['target_modality']['auditory']==int(trial_data['target_modality'][0]):
        modality = 'Auditory Task'
    
    plt.figure()
    plt.text(0,0,subject)
    plt.text(0,-.25,modality)
    plt.ylim([-2,.2])
    plt.xlim([-.05,1])
    plt.xticks([])
    plt.yticks([])
    if save_folder !='':
        plt.savefig(save_folder + subject + "_cover.png")
        plt.close()
        
        
def generate_report(subject, save_folder):
    
    ### cover page    
    
    model_selection_path = glob.glob(save_folder + '*state_number_fitting.png')[0]
    
    state_folders = glob.glob(save_folder + '*states\\')
    for folder in state_folders:
        num_states = os.path.basename(folder[:-2])
        cover_path = (glob.glob(folder + '*_cover.png')[0])
        filepaths = [cover_path,model_selection_path]
        
        imgs = []
        x= []
        y= []
        for filepath in filepaths:
            this_image = plt.imread(filepath)[:,:,:3]
            imgs.append(this_image)
            y.append(this_image.shape[0])
            x.append(this_image.shape[1])
        
        
        page_x = max([sum(x[:]),sum(x[:])])
        page_y = max(y[:])
        cover_page =  np.full((page_y,page_x,3),1,dtype=float)
        
        cum_x = 0
        cum_y = max(y[:])
        for i, img in enumerate(imgs):
            if i==3:
                cum_x=0
            if i<3:
                cover_page[:y[i],cum_x:sum(x[:i+1]),:]=img
                cum_x = x[i]+cum_x
            else:
                cover_page[cum_y:cum_y+y[i],cum_x:sum(x[3:i+1]),:]=img
                cum_x = x[i]+cum_x
        
        cover_page = Image.fromarray(np.uint8(cover_page*255)).convert('RGB')
        
        #### generate page 1 - overview
        page_list = []
    
        image_ids =['*choice_by_state.png', '*state_occupancy.png','*transition_matrix.png',
                    '*GLM_Weights*.png', '*state_dwelltime_distribution.png', '*state_occupancy_by_session.png']
        
        filepaths = []
        for file_ids in image_ids:
            filepaths.append(glob.glob(folder + file_ids)[0])
        
        imgs = []
        x= []
        y= []
        for filepath in filepaths:
            this_image = plt.imread(filepath)[:,:,:3]
            imgs.append(this_image)
            y.append(this_image.shape[0])
            x.append(this_image.shape[1])
        
        
        page_x = max([sum(x[:3]),sum(x[3:])])
        page_y = sum([max(y[:3]),max(y[3:])])
        page1 =  np.full((page_y,page_x,3),1,dtype=float)
        
        cum_x = 0
        cum_y = max(y[:3])
        for i, img in enumerate(imgs):
            if i==3:
                cum_x=0
            if i<3:
                page1[:y[i],cum_x:sum(x[:i+1]),:]=img
                cum_x = x[i]+cum_x
            else:
                page1[cum_y:cum_y+y[i],cum_x:sum(x[3:i+1]),:]=img
                cum_x = x[i]+cum_x
            
        page1_im = Image.fromarray(np.uint8(page1*255)).convert('RGB')
        page_list.append(page1_im)
        
        #### generate page 2 - state/measure relationship
        image_ids =['*pupil_diameter_by_state.png', '*whisker_energy_by_state.png',
                    '*running_speed_by_state.png','*reaction_time_by_state.png', 
                    '*pupil_diameter_stats.png', '*_whisk_trials_violin.png',
                    '*_running_trials_violin.png', '*reaction_time_stats.png',]
        
        filepaths = []
        for file_ids in image_ids:
            filepaths.append(glob.glob(folder + file_ids)[0])
        
        imgs = []
        x= []
        y= []
        for filepath in filepaths:
            this_image = plt.imread(filepath)[:,:,:3]
            imgs.append(this_image)
            y.append(this_image.shape[0])
            x.append(this_image.shape[1])
        
        
        page_x = max([sum(x[:4]),sum(x[4:])])
        page_y = sum([max(y[:4]),max(y[4:])])
        page2 =  np.full((page_y,page_x,3),1,dtype=float)
        
        cum_x = 0
        cum_y = max(y[:4])
        for i, img in enumerate(imgs):
            if i==4:
                cum_x=0
            if i<4:
                page2[:y[i],cum_x:sum(x[:i+1]),:]=img
                cum_x = x[i]+cum_x
            else:
                page2[cum_y:cum_y+y[i],cum_x:sum(x[4:i+1]),:]=img
                cum_x = x[i]+cum_x
                
                
        page2_im = Image.fromarray(np.uint8(page2*255)).convert('RGB')
        page_list.append(page2_im)
        
        ### generate page 3 - state by measure quantiles
        image_ids =['*pupil_diameter_quantile.png', '*whisker_energy_quantile.png',
                '*running_speed_quantile.png']
        filepaths = []
        for file_ids in image_ids:
            filepaths.append(glob.glob(folder + file_ids)[0])
        
        imgs = []
        x= []
        y= []
        for filepath in filepaths:
            this_image = plt.imread(filepath)[:,:,:3]
            imgs.append(this_image)
            y.append(this_image.shape[0])
            x.append(this_image.shape[1])
        
        
        page_x = max([sum(x[:]),sum(x[:])])
        page_y = max(y[:])
        page3 =  np.full((page_y,page_x,3),1,dtype=float)
        
        cum_x = 0
        cum_y = max(y[:])
        for i, img in enumerate(imgs):
            if i==3:
                cum_x=0
            if i<3:
                page3[:y[i],cum_x:sum(x[:i+1]),:]=img
                cum_x = x[i]+cum_x
            else:
                page3[cum_y:cum_y+y[i],cum_x:sum(x[3:i+1]),:]=img
                cum_x = x[i]+cum_x
        
        page3_im = Image.fromarray(np.uint8(page3*255)).convert('RGB')
        page_list.append(page3_im)

        
        ### generate additional pages - session summaries
        measurepaths = glob.glob(folder + '*_stats_v2.png')
        for filepath in measurepaths:
            img = Image.open(filepath).convert('RGB')
            page_list.append(img)
            
        measurespacepaths = glob.glob(folder + '*_measures_space_by_state.png')
        img = Image.open(measurespacepaths[0]).convert('RGB')
        page_list.append(img)    
        
        filepaths = glob.glob(folder + 'line summaries\\*')
        for filepath in filepaths:
            img = Image.open(filepath).convert('RGB')
            page_list.append(img)
        
        ### save pdf
        pdf1_filename = save_folder + subject + '_' + num_states + '_report.pdf'
        # page1_im.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True)
        cover_page.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True, append_images=page_list)

    