import os
import numpy as np
import glob
import pandas as pd
# from scipy import stats
from uobrainflex.behavioranalysis import flex_hmm
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt

cols = ['#ff7f00', '#4daf4a', '#377eb8', '#7E37B8','m','r','c','#008391','#f99f88','k','k','k']

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]

base_folder = 'D:\\Hulsey\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')
hmm_paths = glob.glob(base_folder + 'hmms\\*hmm.npy')

measures_to_use = ['post_hoc_pupil_diameter', 'delta_opt_pupil', 'post_hoc_pupil_std10',
                                    'face_energy',  'whisker_energy', 'face_std_10',
                                    'running_speed', 'running_speed', 'running_std10'] 


mean_measure = np.full([len(hmm_trials_paths),6,len(measures_to_use)],np.nan)
median_measure = np.full([len(hmm_trials_paths),6,len(measures_to_use)],np.nan)

subject_states = {}
subjects=[]
norm_pupil_hist=np.full([len(hmm_trials_paths),6,50],np.nan)
state_probs = np.full([len(hmm_trials_paths),3,50],np.nan)
state_probs_whisk = np.full([len(hmm_trials_paths),3,20],np.nan)
state_probs_run = np.full([len(hmm_trials_paths),3,40],np.nan)
for m in range(len(hmm_trials_paths)):
    print(str(m+1) + ' of ' + str(len(hmm_paths)))
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
    subject = os.path.basename(hmm_paths[m])[:5]
    subjects.append(subject)
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
    
    trial_date=[]
    for trials in hmm_trials:
        trial_date.append(trials['file_name'].iloc[0])
    
    hmm_trials=hmm_trials[np.argsort(trial_date)]        
        
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
        
    opt_pupil = flex_hmm.get_optimal_pupil(hmm_trials)[0] #optimal pupil by [median, polyfit]
    all_trials['delta_opt_pupil']=abs(all_trials['post_hoc_pupil_diameter']-opt_pupil)
        
    for zz, measure in enumerate(measures_to_use):
        
        for state in np.unique(all_trials['hmm_state'].dropna()):
            state_trials = all_trials.query('hmm_state==@state')
            data = state_trials[measure].dropna().values
            mean_measure[m,int(state),zz]=data.mean()
            median_measure[m,int(state),zz]=np.median(data)
            
        state_trials = all_trials.query('hmm_state>1')

        data = state_trials[measure].dropna().values
        mean_measure[m,-1,zz]=data.mean()
        median_measure[m,-1,zz]=np.median(data)

        
lims=[[0,.25],[0,.11],[0,.75],[0,.15],[0,.4],[0,.15]]
sbplt=[1,4,2,5,3,6]
ylbl = ['Measure value \n\nPupil diameter\n(Δ Optimal, % max)','Measure variability \n\nPupil diameter\n10 trial std. (% max)',
        'Face m.e. (% max)','Face m.e.\n10 trial std. (% max)',
        'Locomotion speed (m/s)','Locomotion speed\n10 trial std. (m/s)']
plt.figure(figsize=[16,8])
bplot=[]
ax=[]
for z,measure in enumerate([1,2,3,5,7,8]):
    ax.append(plt.subplot(2,3,sbplt[z]))

    data = [mean_measure[:,1,measure],mean_measure[:,0,measure],mean_measure[:,2,measure]]
    bplot.append(plt.boxplot(data,positions=[0,1,2],showfliers=False,patch_artist=True))

    data = np.vstack(data)
    for this_mouse in data.T:
        x=np.arange(3)+np.random.randn(3)*.05
        plt.plot(x,this_mouse,color=[.7,.7,.7],zorder=10)
        plt.plot(x,this_mouse,'ko',zorder=11)
    plt.ylim(lims[z])
        
    if z==0:
        plt.yticks(np.arange(0,.3,.05),np.arange(0,30,5))
    elif z==1:
        plt.yticks(np.arange(0,.11,.025),np.arange(0,11,2.5))
    elif z==2:
        plt.yticks(np.arange(0,.81,.2),np.arange(0,81,20))
    elif z==3:
        plt.yticks(np.arange(0,.16,.05),np.arange(0,16,5))
    elif z==4:
        plt.yticks(np.arange(0,.41,.1))
    elif z==5:
        plt.yticks(np.arange(0,.16,.05))
    
    plt.xticks([0,1,2],['dis','opt','sub'])
    plt.ylabel(ylbl[z])

ax[0].set_title('Pupil diameter')
ax[2].set_title('Face motion energy')
ax[4].set_title('Locomotion speed')

ax[0].plot([.05,.95],[.22,.22],'k')
ax[0].text(.5,.22,"* * *",ha='center',va='bottom')
ax[0].plot([1.05,1.95],[.22,.22],'k')
ax[0].text(1.5,.22,"* * *",ha='center',va='bottom')

ax[1].plot([.05,.95],[.102,.102],'k')
ax[1].text(.5,.102,'* * *',ha='center',va='bottom')
ax[1].plot([1.05,1.95],[.102,.102],'k')
ax[1].text(1.5,.102,'* * *',ha='center',va='bottom')

ax[2].plot([1.05,1.95],[.7,.7],'k')
ax[2].text(1.5,.7,"*",ha='center',va='bottom')

ax[3].plot([.05,.95],[.14,.14],'k')
ax[3].text(.5,.14,'* *',ha='center',va='bottom')
ax[3].plot([1.05,1.95],[.14,.14],'k')
ax[3].text(1.5,.14,'*',ha='center',va='bottom')

ax[5].plot([.05,.95],[.14,.14],'k')
ax[5].text(.5,.14,'* *',ha='center',va='bottom')
ax[5].plot([1.05,1.95],[.14,.14],'k')
ax[5].text(1.5,.14,'*',ha='center',va='bottom')



for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)
    

a = '#679292'
b ='#c41111'
c ='#ffc589'
colors = [b,a,c]
for b in bplot:
    for patch, color in zip(b['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0)
    for line, color in zip(b['medians'], ['k','k','k']):
        line.set_color(color)
        line.set_linewidth(3)
    for line in b['caps']:
        line.set_color('k')
        line.set_linewidth(2)
    for line in b['whiskers']:
        line.set_color('k')
        line.set_linewidth(2)   

plt.tight_layout()


plt.savefig(base_folder + 'figures\\figure_S4.pdf')
plt.savefig(base_folder + 'figures\\figure_S4.png', dpi=300)