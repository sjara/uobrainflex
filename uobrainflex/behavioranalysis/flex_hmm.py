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

def format_choice_behavior_hmm(trial_data, trial_labels, drop_no_response=True, drop_distractor_trials=True):
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
        
    # trial_start = trials['start_time'].values
    # # create target port ind where left = -1 and right = 1
    # target_port = trials['target_port']
    # target_port = target_port.replace(left_stim,-1)
    # target_port = target_port.replace(right_stim,1)
    # target_port = target_port.values
    
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
    # option output of auditory/visual stim values is possible
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


def compile_choice_hmm_data(sessions,get_behavior_measures = False):
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
        nwbFileObj = load.load_nwb_file(sess)
        trial_data, trial_labels = load.read_trial_data(nwbFileObj)
        if get_behavior_measures:
            behavior_measures = load.read_behavior_measures(nwbFileObj)
            trial_data = load.fetch_trial_beh_measures(trial_data, behavior_measures)
        these_inpts, these_true_choices, trials = format_choice_behavior_hmm(trial_data, trial_labels)
        inpts.extend(these_inpts)
        true_choices.extend([these_true_choices])
        hmm_trials.extend([trials])
        
    return inpts, true_choices, hmm_trials

def choice_hmm_sate_fit(subject, inpts, true_choices, max_states = 4,save_folder=''):
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

def choice_hmm_fit(subject, num_states, inpts, true_choices, hmm_trials):
    ## Fit GLM-HMM with MAP estimation:
    # Set the parameters of the GLM-HMM
    # num_states = 3        # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions


    
    # Instantiate GLM-HMM and set prior hyperparameters (MAP version)
    prior_sigma = 2
    prior_alpha = 2
    
    map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                         observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                         transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    
    N_iters = 200
    fit_map_ll = map_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters,
                                tolerance=10**-4)

    posterior_probs = [map_glmhmm.expected_states(data=data, input=inpt)[0]
                   for data, inpt
                   in zip(true_choices, inpts)]
    
    
    
    for sess_ind in range(len(hmm_trials)):
        session_posterior_prob = posterior_probs[sess_ind] 
        np.max(session_posterior_prob,axis=1)
        trial_max_prob = np.max(session_posterior_prob, axis = 1)
        trial_state = np.argmax(session_posterior_prob, axis = 1).astype(float)
        trial_state[np.where(trial_max_prob<.8)[0]] = np.nan
        hmm_trials[sess_ind]['hmm_state'] = trial_state

    hmm = map_glmhmm
    return hmm, hmm_trials

def plot_GLM_weights(subject, hmm,save_folder=''):
    ## plot results
    cols = ['#ff7f00', '#4daf4a', '#377eb8', 'm','r']
    # Plot covariate GLM weights by HMM state
    plt.figure()
    params = hmm.observations.params
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.ylim([-5, 5])
    for i in range(0,len(params)):
        plt.plot(params[i,:][0]*-1,marker='o',color = cols[i])
        plt.plot(params[i,:][0]*-1,color = cols[i])
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
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
    
def plot_state_occupancy(subject, hmm,true_choices,inpts, save_folder=''):
    # concatenate posterior probabilities across sessions
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
               for data, inpt
               in zip(true_choices, inpts)]
    
    posterior_probs_concat = np.concatenate(posterior_probs)
    # get state with maximum posterior probability at particular trial:
    state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
    # now obtain state fractional occupancies:
    _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
    state_occupancies = state_occupancies/np.sum(state_occupancies)
    
    fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, width = 0.8, color = cols[z])
    plt.ylim((0, 1))
    plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize = 10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.xlabel('state', fontsize = 15)
    plt.ylabel('frac. occupancy', fontsize=15)
    plt.title(subject + ' State Occupancy', fontsize=15)
    if save_folder !='':
        plt.savefig(save_folder + subject + "_state_occupancy.png")
    
    
def plot_state_posteriors_CDF(subject, hmm, inpts, true_choices, save_folder=''):
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
               for data, inpt
               in zip(true_choices, inpts)]
    
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
    cols = ['#ff7f00', '#4daf4a', '#377eb8','m','r']
    posterior_probs = [hmm.expected_states(data=data, input=inpt)[0]
               for data, inpt
               in zip(true_choices, inpts)]
    
    posterior_probs_concat = np.concatenate(posterior_probs)
    # get state with maximum posterior probability at particular trial:
    state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
    num_states =len(hmm.observations.params)

    state_inpt_choice = pd.DataFrame(data=[],index=np.unique(state_max_posterior), columns = np.unique(trial_inpts))
    trial_n = pd.DataFrame(data=[],index=np.unique(state_max_posterior), columns = np.unique(trial_inpts))
    for iState in range(0,num_states):
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
                state_inpt_choice.loc[iState,iInp] = len(np.where(state_choices[these_trials]==1)[0])/len(these_trials)
                trial_n.loc[iState,iInp] =len(these_trials)
    
    
    for iInp in np.unique(trial_inpts):
        these_trials = np.where(trial_inpts == iInp)[0]
        state_inpt_choice.loc['all_trials',iInp] = len(np.where(choice_concat[these_trials]==1)[0])/len(these_trials)
        trial_n.loc['all_trials',iInp] = len(these_trials)
        
    # actual_inpts = [-.98, -.63, -.44, .44, .63, .98]
    # state_inpt_choice.columns = actual_inpts
        
    plt.figure()
    plt.xlabel('Stimulus')
    plt.ylabel('Right Choice Probability')
    plt.title(subject + ' Performance by State')
    for idx in state_inpt_choice.index:
        if idx == 'all_trials':
            plt.plot(state_inpt_choice.iloc[num_states,:],'k--',marker='o')
        if idx != 'all_trials':
            plt.plot(state_inpt_choice.loc[idx,:],marker='o', color=cols[idx])   
    plt.ylim([0,1])
    if save_folder !='':
        plt.savefig(save_folder + subject + "_performance_by_state.png")
    plt.show()  
