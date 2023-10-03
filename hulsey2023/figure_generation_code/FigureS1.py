from pathlib import Path
import numpy as np
import scipy
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt
import pandas as pd
from uobrainflex.nwb import loadbehavior as load
from matplotlib.patches import Rectangle
from scipy import stats

base_folder = input("Enter the main directory path")
folder = Path(x)

r_states=np.full([13,6,4],np.nan)
p_states=np.full([13,6,4],np.nan)
evoked_by_state = np.full([13,6],np.nan)
state_loc_med=np.full([13,4],np.nan)
modality=np.ones(13)
measure_iti_corr = np.full([13,3],np.nan)
evoked_time_lim=2500

#step through data for each mouse
for m,hmm_trials_path in enumerate(list(folder.joinpath('hmm_trials').rglob('*hmm_trials*'))):
    hmm_trials_path = list(folder.joinpath('hmm_trials').rglob('*hmm_trials*'))[m]
    hmm_trials = np.load(hmm_trials_path, allow_pickle = True)
    if np.isnan(hmm_trials[0]['visual_stim_id'].iloc[0]):
        modality[m]=0
        
    start=[]
    evoked_up = []
    evoked_down = []
    hmm_state=[]
    iti=[]
    loc=[]
    face=[]
    # gather data on behavioral measures and ITI
    for trials in hmm_trials:
        for t, pupil in enumerate(trials['pupil_trace']):
            if len(pupil)>1000:
                start.append(pupil[1000])
                evoked_up.append(np.max(pupil[1000:1000+evoked_time_lim] - pupil[1000]))
                evoked_down.append(np.min(pupil[1000:1000+evoked_time_lim] - pupil[1000]))
            else:
                start.append(np.nan)
                evoked_up.append(np.nan)
                evoked_down.append(np.nan)
            
            hmm_state.append(trials['hmm_state'].iloc[t])
            loc.append(trials['running_speed'].iloc[t])
            face.append(trials['face_energy'].iloc[t])
        
        these_iti = np.concatenate([np.array([np.nan]),np.diff(trials['start_time'])]) # prior iti given by difference of start times
        iti.append(these_iti)   
    
    start = np.array(start)
    evoked_up = np.array(evoked_up)
    evoked_down = np.array(evoked_down)
    hmm_state = np.array(hmm_state)
    loc = np.array(loc)
    face = np.array(face)
    iti = np.concatenate(iti)
    
    # drop nans
    nan_ind=[]
    for data in [evoked_up,evoked_down,start,iti,loc,face]:
        nan_ind.append(np.where(data!=data)[0])
        
    nan_ind = list(set(np.concatenate(nan_ind)))
    
    start = np.delete(start,nan_ind)
    evoked_up = np.delete(evoked_up,nan_ind)
    evoked_down = np.delete(evoked_down,nan_ind)
    hmm_state = np.delete(hmm_state,nan_ind)
    iti = np.delete(iti,nan_ind)
    loc = np.delete(loc,nan_ind)
    face = np.delete(face,nan_ind)
    
    # correlate each of the raw measures with the preceding ITI
    for i, measure in enumerate([start,loc,face]):
        this_r, this_p = stats.pearsonr(iti,measure)
        measure_iti_corr[m,i] = this_r

    #determine evoked response
    evoked = np.full(evoked_up.shape[0],np.nan)
    for i in range(len(evoked_up)):
        data = np.array([abs(evoked_up[i]),abs(evoked_down[i])])
        if data.argmax() == 0 :
            evoked[i]=evoked_up[i]
        else:
            evoked[i]=evoked_down[i]          
        
    # get correlations between measures and ITI and evoked pupil split by state
    for i in range(4):
        ind = np.where(hmm_state==i)[0]   
        if i ==2:
            ind = np.where(hmm_state>=i)[0]   
        if i ==3:
            ind = np.where(hmm_state!=hmm_state)[0]   


        ind = ind[:-1]
        if len(ind)>10:
            r_states[m,0,i],p_states[m,0,i]= stats.pearsonr(evoked[ind],start[ind+1])
            r_states[m,1,i],p_states[m,1,i]= stats.pearsonr(evoked[ind],start[ind])
            r_states[m,2,i],p_states[m,2,i]= stats.pearsonr(start[ind],start[ind+1])
            
            r_states[m,3,i],p_states[m,3,i]= stats.pearsonr(start[ind],iti[ind])
            r_states[m,4,i],p_states[m,4,i]= stats.pearsonr(start[ind],iti[ind+1])
            r_states[m,5,i],p_states[m,5,i]= stats.pearsonr(evoked[ind],iti[ind+1])
        evoked_by_state[m,i]=np.nanmedian(evoked[ind]*100)
        state_loc_med[m,i]=np.nanmedian(loc[ind])
        
        
############ plot data
a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f,'k','k','k','k','k','k','k','k']
    
fig = plt.figure(figsize=[16,8])
evoked_time_lim=2500

##### plot data!
#### two example traces of pupil, face energy, and locomotion
for m,lims in zip([4,5],[[2017500,2045000],[3020000,3047500]]):
    hmm_trials_path = list(folder.joinpath('hmm_trials').rglob('*hmm_trials*'))[m]
    hmm_trials = np.load(hmm_trials_path, allow_pickle = True)
    session = 5
    
    file_name = hmm_trials[session]['file_name'].iloc[0]
      
    # find NWB file related to session
    nwb_file_obj = load.load_nwb_file(str(list(folder.rglob(file_name))[0]))
    trial_data,_ = load.read_trial_data(nwb_file_obj)
    # load physiological measures
    behavior_measures = load.read_behavior_measures(nwb_file_obj,measures = 
            ['post_hoc_pupil_diameter','face_energy','running_speed'])
    
    if m==4:
        ax = plt.subplot(2,2,1)
    elif m==5:
        ax = plt.subplot(2,2,3)

    # plot physiological measures
    for measure in ['post_hoc_pupil_diameter','face_energy','running_speed']:
        plt.plot(behavior_measures[measure].iloc[lims[0]:lims[1]])
    
    # calculate and plot evoked pupil along with raw measures at start of trial
    start_times = hmm_trials[session]['start_time'].to_numpy()*1000
    these_starts = start_times[(start_times>lims[0]) & (start_times<lims[1])]
    for start in these_starts:
        start = int(start)
        pupil = behavior_measures['post_hoc_pupil_diameter'].iloc[start]    
        face = behavior_measures['face_energy'].iloc[start]    
        loc = behavior_measures['running_speed'].iloc[start]    
        
        up_evoked_pupil = np.max(behavior_measures['post_hoc_pupil_diameter'].iloc[start:start+evoked_time_lim])
        up_evoked_time = np.argmax(behavior_measures['post_hoc_pupil_diameter'].iloc[start:start+evoked_time_lim])
        
        down_evoked_pupil = np.min(behavior_measures['post_hoc_pupil_diameter'].iloc[start:start+evoked_time_lim])
        down_evoked_time = np.argmin(behavior_measures['post_hoc_pupil_diameter'].iloc[start:start+evoked_time_lim])
    
        if abs(pupil-up_evoked_pupil)>pupil-down_evoked_pupil:
            evoked_pupil = up_evoked_pupil
            evoked_time = up_evoked_time
        else:
            evoked_pupil = down_evoked_pupil
            evoked_time = down_evoked_time        
        
        ax.add_patch(Rectangle([start,-.1], 1200, 1.1,color=[.7,.7,.7],alpha=.2,clip_on=False,edgecolor = None,linewidth=0))
        plt.plot([start,start]+evoked_time,[pupil,evoked_pupil],'k')
        plt.plot([start,start+evoked_time],[pupil,pupil],'k:')
        plt.plot(start,pupil,'ko',markersize=10)
        plt.plot(start,face,'ks',markersize=10)
        plt.plot(start,loc,'k>',markersize=10)
    
    
    
    if m==4:
        plt.text(lims[0]+16500,0.95,'evoked\npupil response',color='k',va='center',ha='center')
        plt.arrow(lims[0]+16500,.87,-900,-.1,
                  length_includes_head=True,width=.02,head_length=300,
                  color='k')
    
    #plot legend
    if m==5:
        plt.plot([lims[0]+14500,lims[0]+19500],[.85,.85],'k',linewidth=5,clip_on=False)
        plt.text(lims[0]+17000,.825,'5 seconds',color='k',va='top',ha='center')
        
        plt.text(lims[0]+12500,1.1725,'Pupil diameter (%max)',color='C0',va='center',ha='right')
        plt.text(lims[0]+12500,1.075,'Face motion energy (a.u.)',color='C1',va='center',ha='right')
        plt.text(lims[0]+15000,1.075,'Locomotion speed (cm/s)',color='C2',va='center',ha='left')
        plt.text(lims[0]+15000,1.1725,'Stimulus',color='k',va='center',ha='left')
        ax.add_patch(Rectangle([lims[0]+13500,1.14], 1200, 0.071,color=[.7,.7,.7],alpha=.2,clip_on=False,edgecolor = None,linewidth=0))

        
        plt.plot([these_starts[0],these_starts[0]],[.825,.875],'k')
        plt.plot([these_starts[1],these_starts[1]],[.825,.875],'k')
        plt.plot([these_starts[0],these_starts[1]],[.85,.85],'k')
        plt.text(int(these_starts[0]+(these_starts[1]-these_starts[0])/2),.825,
                 'Inter trial\ninterval',ha='center',va='top')
        
        plt.plot(lims[0]+2150,1.1725,'ko',markersize=10,clip_on=False)
        plt.plot(lims[0]+1000,1.075,'ks',markersize=10,clip_on=False)
        plt.plot(lims[0]+14250,1.075,'k>',markersize=10,clip_on=False)
                
    plt.yticks(np.arange(0,1.01,.2),np.arange(0,101,20))
    plt.ylim([-.1,1])
    plt.xlim(lims)
    plt.xticks([])

############ plot example ITI pupil correlation
# gather example data
m=4
hmm_trials_path = list(folder.joinpath('hmm_trials').rglob('*hmm_trials*'))[m]
hmm_trials = np.load(hmm_trials_path, allow_pickle = True)
    
start=[]
evoked_up = []
evoked_down = []
hmm_state=[]
iti=[]
loc=[]
for trials in hmm_trials:
    for t, pupil in enumerate(trials['pupil_trace']):
        if len(pupil)>1000:
            start.append(pupil[1000])
            evoked_up.append(np.max(pupil[1000:] - pupil[1000]))
            evoked_down.append(np.min(pupil[1000:] - pupil[1000]))
        else:
            start.append(np.nan)
            evoked_up.append(np.nan)
            evoked_down.append(np.nan)
        
        hmm_state.append(trials['hmm_state'].iloc[t])
        loc.append(trials['running_speed'].iloc[t])
    
    these_iti = np.concatenate([np.array([np.nan]),np.diff(trials['start_time'])])
    iti.append(these_iti)
    

start = np.array(start)
evoked_up = np.array(evoked_up)
evoked_down = np.array(evoked_down)
hmm_state = np.array(hmm_state)
loc = np.array(loc)
iti = np.concatenate(iti)

nan_ind=[]
for data in [evoked_up,evoked_down,start,iti]:
    nan_ind.append(np.where(data!=data)[0])
    
nan_ind = list(set(np.concatenate(nan_ind)))

start = np.delete(start,nan_ind)
evoked_up = np.delete(evoked_up,nan_ind)
evoked_down = np.delete(evoked_down,nan_ind)
hmm_state = np.delete(hmm_state,nan_ind)
iti = np.delete(iti,nan_ind)
loc = np.delete(loc,nan_ind)

evoked = np.full(evoked_up.shape[0],np.nan)
for i in range(len(evoked_up)):
    data = np.array([abs(evoked_up[i]),abs(evoked_down[i])])
    if data.argmax() == 0 :
        evoked[i]=evoked_up[i]
    else:
        evoked[i]=evoked_down[i]
        
# plot example correlation between ITI and pupil
plt.subplot(2,4,3)
plt.plot(iti,start,'o',color=[.7,.7,.7],markersize=1)

coeffs = np.polyfit(iti,start,1)
this_poly = np.poly1d(coeffs)
plt.plot([min(iti),max(iti)],[this_poly(min(iti)),this_poly(max(iti))],'k')
plt.xlim(0,100)
r,p = stats.pearsonr(iti,start)


plt.text(60,.9,'r = {:.2f}'.format(r))
plt.xlabel('Prior inter tiral interval (s)')
plt.ylabel('Pupil diameter (%max)')
plt.yticks(np.arange(0.4,1.01,.1),[40,'',60,'',80,'',100])
plt.ylim(.31,1)


# plot group data for measure/ITI correlation, Evoked responses by state and evoked vs next pupil relation by state
titles=['Measure vs prior ITI', 'Evoked pupil response (% max)', 'Evoked response vs next pupil']
ylabels=['Measure/prior ITI correlation (r)','Evoked pupil response (% max)','Evoked pupil/next pupil correlation (r)']
for data,i,ylbl in zip([measure_iti_corr,evoked_by_state[:,:4],r_states[:,0,:]],[4,7,8],ylabels):
    plt.subplot(2,4,i)
 
    if i != 7:
        plt.ylim([-1,1])
    plt.ylabel(ylbl)
    if i!=4:
        bplot = plt.boxplot(data,showfliers=False,patch_artist=True)
        plt.xticks(range(1,5),['Opt','Dis','Sub','Ind'])
        plt.plot([.5,4.5],[0,0],'k--')
        for d in data:
            x=range(1,len(d)+1)+np.random.randn(len(d))/20
            plt.plot(x,d,'ko',markersize=4,zorder=11)
            plt.plot(x,d,color=[.5,.5,.5],zorder=10,linewidth=1)
        plt.xlabel('Performance State')
    else:
        bplot = plt.boxplot(data,showfliers=False,patch_artist=True)
        for d in data:
            plt.plot(range(1,len(d)+1)+np.random.randn(len(d))/20,d,'ko',markersize=4,zorder=10)
        plt.xticks(range(1,4),['Pupil','Face m.e.','Loc. speed'])
        plt.plot([.5,3.5],[0,0],'k--')
        plt.xlabel('Measure')
    
    a = '#679292'
    b ='#c41111'
    c ='#ffc589'
    colors = [a,b,c,[.7,.7,.7]]
    if i==4:
        colors=['C0','C1','C2']
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)
    for line in bplot['medians']:
        line.set_color('k')
        line.set_linewidth(3)
    for line in bplot['caps']:
        line.set_color('k')
        line.set_linewidth(2)
    for line in bplot['whiskers']:
        line.set_color('k')
        line.set_linewidth(2) 
    

plt.tight_layout()

for spine in ['top','right']:
    for axs in fig.axes:
        axs.spines[spine].set_visible(False)
        

for spine in ['bottom']:
    for axs in fig.axes[:2]:
        axs.spines[spine].set_visible(False)
        
# plt.savefig(r'D:\Hulsey\Hulsey_et_al_2023\figures\Figure_ITI_Supplement.pdf')
        
        