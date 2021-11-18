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
            
    ## build stim values for inpt
    # build arrays for visual and auditory stimulus identities
    vis_dir = np.zeros(len(trials))    
    vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['left'])[0]]=-1
    vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['right'])[0]]=1
    vis_dir[np.where(trials['visual_stim_id']==trial_labels['visual_stim_id']['no stim'])[0]]=np.nan
    
    aud_dir = np.zeros(len(trials))
    aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['left'])[0]]=-1
    aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['right'])[0]]=1
    aud_dir[np.where(trials['auditory_stim_id']==trial_labels['auditory_stim_id']['no stim'])[0]]=np.nan
  
    # get array of auditory values
    aud_val = trials['auditory_stim_difficulty'].values

    # get visual gabor angle if available. if not available, stim values are hard coded as appropriate
    vis_val = trials['visual_stim_difficulty'].values
    vis_stim_id = np.zeros(len(trials))
    if any(trials.columns.values==['visual_gabor_angle']):
        stim_val = trials['visual_gabor_angle'].values
        stim_val = stim_val[~np.isnan(stim_val)]
        stim_val = ((stim_val - 45)/45)
        stim_val = np.flip(np.unique(abs(stim_val)))
        for this_diff in np.flip(np.unique(vis_val)):
            ind = np.where(vis_val==this_diff)[0]
            vis_stim_id[ind]=stim_val[int(this_diff)]
    else:
        vis_stim_id[vis_val==1]=.8
        vis_stim_id[vis_val==2]=.6
        vis_stim_id[vis_val==0]=1
    
    # this assumes target modality is constant, and will only output stim vals for the target modality
    # getting both values will be necessary for models incorporating distractors
    if trials.iloc[0]['target_modality']==trial_labels['target_modality']['auditory']:
        stim = aud_val*aud_dir
    elif trials.iloc[0]['target_modality']==trial_labels['target_modality']['visual']:
        stim = vis_stim_id*vis_dir
    
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
        nwbFileObj = load.load_nwb_file(sess)
        trial_data, trial_labels = load.read_trial_data(nwbFileObj)
        if get_behavior_measures:
            behavior_measures = load.read_behavior_measures(nwbFileObj)
            _ ,behavior_events = load.read_behavior_events(nwbFileObj)
            trial_data = load.fetch_trial_beh_measures(trial_data, behavior_measures)
            trial_data = load.fetch_trial_beh_events(trial_data, behavior_events)
        these_inpts, these_true_choices, trials = format_choice_behavior_hmm(trial_data, trial_labels)
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
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions
    
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
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#7E37B8']
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
    plt.show()
    plt.title(subject + ' Model Fitting')
    if save_folder !='':
        plt.savefig(save_folder + subject +'_state_number_fitting.png')
    
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
    # num_states = 3        # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(true_choices)))    # number of categories for output
    input_dim = inpts[0].shape[1]                                    # input dimensions

    # Instantiate GLM-HMM and set prior hyperparameters (MAP version)
    prior_sigma = 2
    prior_alpha = 2
    
    map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                         observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                         transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    
    N_iters = 200
    fit_map_ll = map_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters,
                                tolerance=10**-4)
    hmm = map_glmhmm
    return hmm

def get_posterior_probs(hmm, true_choices, inpts, hmm_trials, occ_thresh = .8):
    #pass true data to the model and get state occupancy probabilities
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
                   for data, inpt
                   in zip(true_choices, inpts)]
    
    #update hmm_trials with state probabilities
    for sess_ind in range(len(hmm_trials)):
        session_posterior_prob = posterior_probs[sess_ind] 
        np.max(session_posterior_prob,axis=1)
        trial_max_prob = np.max(session_posterior_prob, axis = 1)
        trial_state = np.argmax(session_posterior_prob, axis = 1).astype(float)
        trial_state[np.where(trial_max_prob<occ_thresh)[0]] = np.nan
        hmm_trials[sess_ind]['hmm_state'] = trial_state
        for state in range(session_posterior_prob.shape[1]):
            hmm_trials[sess_ind]['state_' + str(state+1) +'_prob'] = session_posterior_prob[:,state]

    return posterior_probs, hmm_trials

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

    cols = ['#ff7f00', '#4daf4a', '#377eb8', 'm','r']
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


def plot_session_posterior_probs(subject, hmm,true_choices,inpts, save_folder=''):
    # plot trial state probability by session
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
                       for data, inpt
                       in zip(true_choices, inpts)]
    
    num_states =len(hmm.observations.params)
    
    for sess_id in range(0,len(true_choices)):
        plt.figure()
        cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']
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
        plt.show()

def plot_transition_matrix(subject, hmm, save_folder=''):
    # plot state transition matrix
    num_states =len(hmm.observations.params)
    plt.figure()
    recovered_trans_mat = np.exp(hmm.transitions.log_Ps)
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
    plt.title("transition matrix", fontsize = 20)
    if save_folder!='':
        plt.savefig(save_folder + subject +" transition_matrix.png")


def plot_state_occupancy(subject, hmm_trials, save_folder=''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']

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

def plot_state_posteriors_CDF(subject, posterior_probs, save_folder=''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']
    
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


def plot_state_psychometrics(subject, hmm, inpts,true_choices,save_folder=''):
    #generate psychometrics per state
    #concatenate inpts and choices
    
    inpts_concat = np.concatenate(inpts)
    trial_inpts = inpts_concat[:,0]
    choice_concat = np.concatenate(true_choices)
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','k']
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
               for data, inpt
               in zip(true_choices, inpts)]
    
    posterior_probs_concat = np.concatenate(posterior_probs)
    # get state with maximum posterior probability at particular trial:
    ind = np.where(posterior_probs_concat>.8)[0]
    state =  np.where(posterior_probs_concat>.8)[1]
    state_max_posterior = np.full(len(posterior_probs_concat),np.nan)
    state_max_posterior[ind] = state
    # state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
    num_states = hmm.observations.params.shape[0]

    state_idx = np.unique(state_max_posterior)
    state_idx = state_idx[~np.isnan(state_idx)]
    state_inpt_l_choice = pd.DataFrame(data=[],index=state_idx, columns = np.unique(trial_inpts))
    state_inpt_r_choice = pd.DataFrame(data=[],index=state_idx, columns = np.unique(trial_inpts))
    state_inpt_n_choice = pd.DataFrame(data=[],index=state_idx, columns = np.unique(trial_inpts))
    trial_n = pd.DataFrame(data=[],index=state_idx, columns = np.unique(trial_inpts))
    for iState in range(0,num_states):
            state_ind = np.where(state_max_posterior == iState)[0].astype(int)
            posterior_probs_concat>.8
            state_ind = np.where(state_max_posterior == iState)[0].astype(int)
            state_inpts = trial_inpts[state_ind]
            state_choices = choice_concat[state_ind]
            for iInp in np.unique(state_inpts):
                these_trials = np.where(state_inpts == iInp)[0]
                #if iInp > 0:
                    #correct_choice = 1
                    #mult=1
                #if iInp <0:
                    #correct_choice = 0
                    #mult=-1
                #state_inpt_choice.loc[iState,iInp] = len(np.where(state_choices[these_trials]==correct_choice)[0])/len(these_trials)*mult
                state_inpt_l_choice.loc[iState,iInp] = len(np.where(state_choices[these_trials]==0)[0])/len(these_trials)
                state_inpt_r_choice.loc[iState,iInp] = len(np.where(state_choices[these_trials]==1)[0])/len(these_trials)
                state_inpt_n_choice.loc[iState,iInp] = len(np.where(state_choices[these_trials]==2)[0])/len(these_trials)
                trial_n.loc[iState,iInp] =len(these_trials)
    
    
    for iInp in np.unique(trial_inpts):
        these_trials = np.where(trial_inpts == iInp)[0]
        state_inpt_l_choice.loc['all_trials',iInp] = len(np.where(choice_concat[these_trials]==0)[0])/len(these_trials)
        state_inpt_r_choice.loc['all_trials',iInp] = len(np.where(choice_concat[these_trials]==1)[0])/len(these_trials)
        state_inpt_n_choice.loc['all_trials',iInp] = len(np.where(choice_concat[these_trials]==2)[0])/len(these_trials)
        trial_n.loc['all_trials',iInp] = len(these_trials)
        
        
    columns = state_inpt_l_choice.columns
    index = state_inpt_l_choice.index
    hcols = int(len(columns)/2)
    state_inpt_hits = pd.DataFrame(data=[],index=index, columns = columns)
    state_inpt_errors = pd.DataFrame(data=[],index=index, columns = columns)
    state_inpt_misses = pd.DataFrame(data=[],index=index, columns = columns)
    
    state_inpt_hits.iloc[:,:hcols] = state_inpt_l_choice.iloc[:,:hcols]
    state_inpt_hits.iloc[:,hcols:] = state_inpt_r_choice.iloc[:,hcols:]
    
    state_inpt_errors.iloc[:,:hcols] = state_inpt_r_choice.iloc[:,:hcols]
    state_inpt_errors.iloc[:,hcols:] = state_inpt_l_choice.iloc[:,hcols:]
    state_inpt_misses.iloc[:,:] = state_inpt_n_choice.iloc[:,:]
        
    # actual_inpts = [-.98, -.63, -.44, .44, .63, .98]
    # state_inpt_choice.columns = actual_inpts
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','k']

    # states = state_inpt_hits.index
    # for idx,state in enumerate(states):
    #     plt.plot(state_inpt_hits.loc[state,:],color=cols[idx],linewidth=3,marker='o')    
    #     plt.plot(state_inpt_errors.loc[state,:],'--',color=cols[idx])    
    #     plt.plot(state_inpt_misses.loc[state,:],':',color=cols[idx])    
    
    if len(np.unique(choice_concat))<3:
        plt.figure()
        plt.xlabel('Stimulus')
        plt.ylabel('Right Choice Probability')
        plt.title(subject + ' Performance by State')
        for idx in state_inpt_r_choice.index:
            if idx == 'all_trials':
                plt.plot(state_inpt_r_choice.columns,state_inpt_r_choice.loc[idx,:].values,'k--',marker='o')
            if idx != 'all_trials':
                plt.plot(state_inpt_r_choice.columns,state_inpt_r_choice.loc[idx,:].values,marker='o', color=cols[idx])   
    elif len(np.unique(choice_concat))==3:
        states = state_inpt_hits.index
        n_columns = round(len(states)/2)+len(states)%2
        
        plt.figure(figsize=(4*n_columns, 6))
        for idx,state in enumerate(states):
            plt.subplot(2, n_columns, idx+1)
            plt.plot(state_inpt_hits.loc[state,:],color=cols[idx],marker='o')    
            plt.plot(state_inpt_errors.loc[state,:],'--',color=cols[idx],marker='^')    
            plt.plot(state_inpt_misses.loc[state,:],':',color=cols[idx],marker='s')   
            if state != 'all_trials':
                plt.title('State ' + str(state+1))
            else:
                plt.title('All Trials')
            plt.legend(['Hit','Error','Miss'])
            plt.ylabel('Probability')
            plt.ylim([-.02,1.02])
            plt.xticks([-1,-.5,0,.5,1])
        plt.suptitle(subject + ' Performance by State',fontsize=20)
        if save_folder !='':
            plt.savefig(save_folder + subject + "_performance_by_state.png")
        states = state_inpt_l_choice.index
        plt.figure(figsize=(4*n_columns, 6))
        for idx,state in enumerate(states): 
            plt.subplot(2, n_columns, idx+1)
            plt.plot(state_inpt_l_choice.loc[state,:],color=cols[idx],marker='<')    
            plt.plot(state_inpt_r_choice.loc[state,:],'--',color=cols[idx],marker='>')    
            plt.plot(state_inpt_n_choice.loc[state,:],':',color=cols[idx],marker='s') 
            if state != 'all_trials':
                plt.title('State ' + str(state+1))
            else:
                plt.title('All Trials')
            plt.legend(['L Choice','R Choice','Miss'])
            plt.ylabel('Probability')
            plt.ylim([-.02,1.02])
            plt.xticks([-1,-.5,0,.5,1])
        plt.suptitle(subject + ' Choice by State',fontsize=20)
        if save_folder !='':
            plt.savefig(save_folder + subject + "_choice_by_state.png")
        
        plt.figure(figsize=(12, 6))
        L = plt.subplot(1, 3, 1)
        for idx, state in enumerate(state_inpt_l_choice.index):
            if state == 'all_trials':
                L.plot(state_inpt_l_choice.columns,state_inpt_l_choice.loc[state,:].values,'k--',marker='o')
            else:
                L.plot(state_inpt_l_choice.columns,state_inpt_l_choice.loc[state,:].values,marker='o', color=cols[idx])
            plt.title('Weights for L')
            plt.ylabel('Probability')
            plt.legend(['State 1','State 2','State 3','All Trials'])
            plt.ylim([-0.01,1.01])
            
        R = plt.subplot(1, 3, 2)
        for idx, state in enumerate(state_inpt_l_choice.index):
            if state == 'all_trials':
                R.plot(state_inpt_r_choice.columns,state_inpt_r_choice.loc[state,:].values,'k--',marker='o')
            else:
                R.plot(state_inpt_r_choice.columns,state_inpt_r_choice.loc[state,:].values,marker='o', color=cols[idx])
            plt.title('Weights for R')
            plt.ylim([-0.01,1.01])

        M = plt.subplot(1, 3, 3)
        for idx, state in enumerate(state_inpt_l_choice.index):
            if state == 'all_trials':
                M.plot(state_inpt_n_choice.columns,state_inpt_n_choice.loc[state,:].values,'k--',marker='o')
            else:
                M.plot(state_inpt_n_choice.columns,state_inpt_n_choice.loc[state,:].values,marker='o', color=cols[idx])
            plt.title('Weights for Miss')
            plt.ylim([-0.01,1.01])
            plt.suptitle(subject + ' Performance by State',fontsize=20)
    if save_folder !='':
        plt.savefig(save_folder + subject + "_performance_weight.png")
    
def plot_dwell_times(subject, hmm_trials, save_folder = ''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8', 'm','r']
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
        
    return session_transitions
        
    
def plot_session_summaries(subject, hmm_trials,nwbfilepaths,save_folder =''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8', 'm','r']

    for session, these_trials in enumerate(hmm_trials):
        columns = these_trials.columns
        # plt.figure()    
        fig = plt.figure(figsize=(25, 12), facecolor='w', edgecolor='k')
        plt.plot(these_trials['state_1_prob'].values,cols[0],linewidth=3)
        plt.plot(these_trials['state_2_prob'].values,cols[1],linewidth=3)
        lg = ['state 1','state 2']
        if any(columns=='state_3_prob'):
            plt.plot(these_trials['state_3_prob'].values,cols[2],linewidth=3)
            lg.append('state 3')
        if any(columns=='state_4_prob'):
            plt.plot(these_trials['state_4_prob'].values,cols[3],linewidth=3)
            lg.append('state 4')
        pupil = these_trials['pupil_size']
        pupil = pupil/2+.5
        plt.scatter(range(len(these_trials)),pupil,color='k')
        run = these_trials['run']
        run = run/max(run)/6+.33
        plt.scatter(range(len(these_trials)),run,color='k',marker = ">")
        whisk = these_trials['whisk']
        whisk = whisk/max(whisk)/6+.16
        plt.scatter(range(len(these_trials)),whisk,color='k',marker = "s")
        rt = these_trials['reaction_time']
        rt = rt/max(rt)/6
        plt.scatter(range(len(these_trials)),rt,color='k',marker = "*")
        for measure in ['pupil','wheel','whisk']:
            lg.append(measure)
        plt.legend(lg)
        plt.xlabel('Trial',fontsize = 30)
        plt.title(nwbfilepaths[session],fontsize=17)
        if save_folder != '':
            plt.savefig(save_folder +  subject + "_session_" + str(session) + "_summary.png")   


def plot_session_summaries_patch(subject, hmm_trials,dwell_times,nwbfilepaths,save_folder =''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8', 'm','r']

    for session, these_trials in enumerate(hmm_trials):
        hand=[]
        lg=[]
        dwells = dwell_times[session]
        fig = plt.figure(figsize=(25, 12), facecolor='w', edgecolor='k')
        ax = fig.add_axes([0.1, .1, 0.8, .8])

        for idx in dwells.index: 
            ax.add_patch(Rectangle((dwells.loc[idx,'first_trial']-.4,0), dwells.loc[idx,'dwell']-1.1, 1,color=cols[int(dwells.loc[idx,'state'])],alpha=.2))
        for idx in range(0,int(max(dwells['state']))+1):
            hand.append(ax.add_patch(Rectangle((0,0),0,0,color=cols[idx],alpha=.2)))
            lg.append('state ' + str(idx+1))
        pupil = these_trials['pupil_size']
        pupil = pupil/2+.5
        hand.append(plt.scatter(range(len(these_trials)),pupil,color='k'))
        run = these_trials['run']
        run = run/max(run)/6+.33
        hand.append(plt.scatter(range(len(these_trials)),run,color='k',marker = ">"))
        whisk = these_trials['whisk']
        whisk = whisk/max(whisk)/6+.16
        hand.append(plt.scatter(range(len(these_trials)),whisk,color='k',marker = "s"))
        rt = these_trials['reaction_time']
        rt = rt/max(rt)/6
        hand.append(plt.scatter(range(len(these_trials)),rt,color='k',marker = "*"))
        for measure in ['pupil','wheel','whisk']:
            lg.append(measure)
        plt.legend(hand,lg)
        plt.xlabel('Trial',fontsize = 30)
        plt.title(nwbfilepaths[session],fontsize=17)
        if save_folder != '':
            plt.savefig(save_folder +  subject + "_session_" + str(session) + "_summary.png")   
            
            
def plot_state_measure_histograms(subject, hmm_trials, save_folder = ''):
    all_pupil = np.empty(0)
    all_run = np.empty(0)
    all_whisk = np.empty(0)
    S1_posterior = np.empty(0)
    S2_posterior = np.empty(0)
    S3_posterior = np.empty(0)
    S4_posterior = np.empty(0)
    
    trial_df = pd.DataFrame()
    for trials in hmm_trials:
        all_pupil = np.append(all_pupil, trials['pupil_size'].values)
        all_run = np.append(all_run, trials['run'].values)
        all_whisk = np.append(all_whisk, trials['whisk'].values)
        S1_posterior = np.append(S1_posterior, trials['state_1_prob'].values)
        S2_posterior = np.append(S2_posterior, trials['state_2_prob'].values)
        S3_posterior = np.append(S3_posterior, trials['state_3_prob'].values)
        S4_posterior = np.append(S4_posterior, trials['state_4_prob'].values)
        
    trial_df['pupil']=all_pupil
    trial_df['whisk']=all_whisk
    trial_df['run']=all_run
    trial_df['state']=np.nan
    trial_df.loc[np.where(S1_posterior>.8)[0],'state']=1
    trial_df.loc[np.where(S2_posterior>.8)[0],'state']=2
    trial_df.loc[np.where(S3_posterior>.8)[0],'state']=3
    trial_df.loc[np.where(S4_posterior>.8)[0],'state']=4
    
    
    S1 = trial_df.loc[trial_df['state'] == 1]
    S2 = trial_df.loc[trial_df['state'] == 2]
    S3 = trial_df.loc[trial_df['state'] == 3]
    S4 = trial_df.loc[trial_df['state'] == 4]
    
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','k']
    ## pupil
    plt.figure()
    S1_data = np.histogram(S1['pupil'],np.arange(0,1,.02))
    S2_data = np.histogram(S2['pupil'],np.arange(0,1,.02))
    S3_data = np.histogram(S3['pupil'],np.arange(0,1,.02))
    S4_data = np.histogram(S4['pupil'],np.arange(0,1,.02))
    plt.plot(S1_data[1][:-1],S1_data[0]/sum(S1_data[0]),color=cols[0],linewidth=3)
    plt.plot(S2_data[1][:-1],S2_data[0]/sum(S2_data[0]),color=cols[1],linewidth=3)
    plt.plot(S3_data[1][:-1],S3_data[0]/sum(S3_data[0]),color=cols[2],linewidth=3)
    plt.plot(S4_data[1][:-1],S4_data[0]/sum(S4_data[0]),color=cols[3],linewidth=3)
    
    plt.legend(['S1, n = ' + str(len(S1)),'S2, n = ' + str(len(S2)),
                'S3, n = ' + str(len(S3)),'S4, n = ' + str(len(S4))])
    
    plt.title(subject + '\nPupil distribution by state')
    plt.xlabel('Pupil Diameter (% max)')
    plt.ylabel('Probability')
    
    if save_folder !='':
        plt.savefig(save_folder + subject + "_pupils_by_state.png")
    ## run
    plt.figure()
    S1_data = np.histogram(S1['run'],np.arange(-.1,.5,.02))
    S2_data = np.histogram(S2['run'],np.arange(-.1,.5,.02))
    S3_data = np.histogram(S3['run'],np.arange(-.1,.5,.02))
    S4_data = np.histogram(S4['run'],np.arange(-.1,.5,.02))
    plt.plot(S1_data[1][:-1],S1_data[0]/sum(S1_data[0]),color=cols[0],linewidth=3)
    plt.plot(S2_data[1][:-1],S2_data[0]/sum(S2_data[0]),color=cols[1],linewidth=3)
    plt.plot(S3_data[1][:-1],S3_data[0]/sum(S3_data[0]),color=cols[2],linewidth=3)
    plt.plot(S4_data[1][:-1],S4_data[0]/sum(S4_data[0]),color=cols[3],linewidth=3)
    
    plt.legend(['S1, n = ' + str(len(S1)),'S2, n = ' + str(len(S2)),
                'S3, n = ' + str(len(S3)),'S4, n = ' + str(len(S4))])
    
    plt.title(subject + '\nWheel speed distribution by state')
    plt.xlabel('Wheel speed (m/s)')
    plt.ylabel('Probability')
    if save_folder !='':
        plt.savefig(save_folder + subject + "_running_by_state.png")
    
    ## whisk
    plt.figure()
    S1_data = np.histogram(S1['whisk'],np.arange(0,.2,.01))
    S2_data = np.histogram(S2['whisk'],np.arange(0,.2,.01))
    S3_data = np.histogram(S3['whisk'],np.arange(0,.2,.01))
    S4_data = np.histogram(S4['whisk'],np.arange(0,.2,.01))
    plt.plot(S1_data[1][:-1],S1_data[0]/sum(S1_data[0]),color=cols[0],linewidth=3)
    plt.plot(S2_data[1][:-1],S2_data[0]/sum(S2_data[0]),color=cols[1],linewidth=3)
    plt.plot(S3_data[1][:-1],S3_data[0]/sum(S3_data[0]),color=cols[2],linewidth=3)
    plt.plot(S4_data[1][:-1],S4_data[0]/sum(S4_data[0]),color=cols[3],linewidth=3)
    
    plt.legend(['S1, n = ' + str(len(S1)),'S2, n = ' + str(len(S2)),
                'S3, n = ' + str(len(S3)),'S4, n = ' + str(len(S4))])
    
    plt.title(subject +'\nWhisker motion energy distribution by state')
    plt.xlabel('Motion Energy (a.u.)')
    plt.ylabel('Probability')
    if save_folder !='':
        plt.savefig(save_folder + subject + "_whisk_by_state.png")