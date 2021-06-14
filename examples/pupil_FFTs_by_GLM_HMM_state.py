# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:11:32 2021

Plot FFTs of pupil during states extraced by GLM_HMM model
"""


from uobrainflex.nwb import loadbehavior as load
from uobrainflex.behavioranalysis import flex_hmm
import ssm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack

subject='BW016'
sessions=[]
for fileID in range(20210421,20210428):
    this_session = load.get_file_path(subject,str(fileID))
    if this_session:
        sessions.append(this_session)


inpts=list([])
true_choices=list([])
sessions_trial_start =list([])
for sess in sessions:
    trial_data, trial_labels = load.load_trial_data(sess)
    these_inpts, these_true_choices, trial_start = flex_hmm.format_choice_behavior_hmm(trial_data, trial_labels)
    inpts.extend(these_inpts)
    true_choices.extend([these_true_choices])
    sessions_trial_start.extend([trial_start])



# Set the parameters of the GLM-HMM
num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions
###### Fit GLM-HMM with MAP estimation:
    
# Instantiate GLM-HMM and set prior hyperparameters (MAP version)
prior_sigma = 2
prior_alpha = 2

map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                     observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                     transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))

N_iters = 200
fit_map_ll = map_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters,
                            tolerance=10**-4)




# Show fit 
params = map_glmhmm.observations.params
cols = ['#ff7f00', '#4daf4a', '#377eb8']
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.ylim([-3.5, 3.5])
for i in range(0,len(params)):
    plt.plot(params[i,:][0]*-1,marker='o',color = cols[i])
    plt.plot(params[i,:][0]*-1,color = cols[i])
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
#plt.savefig("glm_state_weights.svg")
plt.show()




# Get expected states:
posterior_probs = [map_glmhmm.expected_states(data=data, input=inpt)[0]
                   for data, inpt
                   in zip(true_choices, inpts)]

### plot state probability by session
if 1:
    for sess_id in range(0,len(true_choices)):
        plt.clf()
        cols = ['#ff7f00', '#4daf4a', '#377eb8']
        #sess_id = 2 #session id; can choose any index between 0 and num_sess-1
        for indst in range(num_states):
            plt.plot(posterior_probs[sess_id][:, indst], label="State " + str(indst + 1), lw=2,
                     color=cols[indst])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize = 10)
        plt.xlabel("trial #", fontsize = 15)
        plt.ylabel("p(state)", fontsize = 15)
        plt.title('session ' + str(sess_id))
        #plt.savefig(f"Pstate_session{sess_id}.svg")
        plt.show()


### plot state transition matrix
recovered_trans_mat = np.exp(map_glmhmm.transitions.log_Ps)
plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(recovered_trans_mat.shape[0]):
    for j in range(recovered_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=4)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xlabel('state t+1', fontsize = 15)
plt.ylabel('state t', fontsize = 15)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.title("transition matrix", fontsize = 20)
plt.subplots_adjust(0, 0, 1, 1)
#plt.savefig("transition_matrix.svg")

### plot state occupancy
# concatenate posterior probabilities across sessions
posterior_probs_concat = np.concatenate(posterior_probs)
# get state with maximum posterior probability at particular trial:
state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
# now obtain state fractional occupancies:
_, state_occupancies = np.unique(state_max_posterior, return_counts=True)
state_occupancies = state_occupancies/np.sum(state_occupancies)

#plot
fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
for z, occ in enumerate(state_occupancies):
    plt.bar(z, occ, width = 0.8, color = cols[z])
plt.ylim((0, 1))
plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize = 10)
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
plt.xlabel('state', fontsize = 15)
plt.ylabel('frac. occupancy', fontsize=15)
#plt.savefig("state_occupancy.svg")


### generate psychometrics per state

#concatenate inpts and choices
inpts_concat = np.concatenate(inpts)
trial_inpts = inpts_concat[:,0]
choice_concat = np.concatenate(true_choices)

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
    
actual_inpts = [-.98, -.63, -.44, .44, .63, .98]
state_inpt_choice.columns = actual_inpts
    
plt.clf()
plt.xlabel('Stimulus')
plt.ylabel('Right Choice Probability')
plt.title('Performance by State')
for idx in state_inpt_choice.index:
    if idx == 'all_trials':
        plt.plot(state_inpt_choice.iloc[3,:],'k--',marker='o')
    if idx != 'all_trials':
        plt.plot(state_inpt_choice.loc[idx,:],marker='o', color=cols[idx])   
#plt.savefig("performance_by_state.svg")
plt.show()  


############# plot FFT of pupil size vs stage per session
#for ind in range(len(sessions)):
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
cols = ['#ff7f00', '#4daf4a', '#377eb8']
    
for ind in range(len(sessions)): # iterate this for looking across sessions       
    state_times = pd.DataFrame()
    trial_state = np.argmax(posterior_probs[ind], axis = 1)
    trial_start = sessions_trial_start[ind]
    s_ind = 0
    t_ind = 0
    prior_state = np.nan
    while s_ind < len(trial_state):
        current_state = trial_state[s_ind]
        if prior_state != current_state:
            state_times.loc['start_trial',t_ind] = s_ind
            state_times.loc['start_time',t_ind] = trial_start[s_ind]
            state_times.loc['state',t_ind] = current_state
            if t_ind != 0:
                state_times.loc['end_trial',t_ind-1] = s_ind-1
                state_times.loc['end_time',t_ind-1] = trial_start[s_ind-1]
            t_ind=t_ind+1
        s_ind=s_ind+1
        prior_state = current_state
    state_times.loc['end_trial',t_ind-1] = s_ind-1
    state_times.loc['end_time',t_ind-1] = trial_start[s_ind-1]
    state_times.loc['n_trials'] = state_times.loc['end_trial']-state_times.loc['start_trial']
    state_times.loc['duration'] = state_times.loc['end_time']-state_times.loc['start_time']
    
    
    behavior_measures = load.load_behavior_measures(sessions[ind], speriod=0.001)
    idx = behavior_measures.index
    
    for t_ind in range(len(state_times.loc['state'])):
        st = round(state_times.loc['start_time',t_ind]*1000)
        end =round(state_times.loc['end_time',t_ind]*1000)
    
        this_data = behavior_measures.loc[idx[st:end],'pupil_size']
        if len(this_data)>100:
            # Number of samplepoints
            N = len(this_data)
            # sample spacing
            T = 1.0 / 1000
            x = np.linspace(0.0, N*T, N)
            y = np.array(this_data)-np.mean(this_data)  #np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
            yf = scipy.fftpack.fft(y)
            xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
        
            plt.plot(xf, smooth(2.0/N * np.abs(yf[:N//2]),10),color=cols[int(state_times.loc['state',t_ind])])
    plt.title('session ' + str(ind))
    plt.xlim(0, 1)
    plt.xlabel('period (s)', fontsize = 15)
    plt.ylabel('power', fontsize=15)
    plt.show()




