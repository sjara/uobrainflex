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
from scipy import stats

base_folder = input("Enter the main directory path")
folder = Path(base_folder)

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]

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

def compare_psychos(psycho1,psycho2):
    all_similarity=[]    
    for state in psycho2:
        similarity=[]
        for state2 in psycho1:
            similarity.append(np.nansum(abs(state-state2)))
        all_similarity.append(similarity)
    return np.array(all_similarity)

all_r=np.full([13,6],np.nan)
all_states_r = np.full([13,4,3],np.nan)

hmm_trials_paths = list(folder.glob('*hmm_trials*'))
state_probs_move = np.full([len(hmm_trials_paths),6,40],np.nan)
subjects = []
for m in range(len(hmm_trials_paths)):
    hmm_trials_path = list(folder.glob('*hmm_trials*'))[m]
    subjects.append(hmm_trials_paths[m].name[:5])
    
    
    hmm_trials = np.load(hmm_trials_path, allow_pickle = True)
    
    trial_date=[]
    for trials in hmm_trials:
        trial_date.append(trials['file_name'].iloc[0])
    
    hmm_trials=hmm_trials[np.argsort(trial_date)]        
        
    psychos, xvals = get_psychos(hmm_trials)
    sort_psychos = np.zeros([6,3,6])
    #optimal
    sort_psychos[0,0,:3] = 1
    sort_psychos[0,1,3:] = 1
    #disengaged
    sort_psychos[1,2,:] = 1
    #all left
    sort_psychos[2,0,:] = 1
    #left and no
    sort_psychos[3,0,:3] = 1
    sort_psychos[3,2,3:] = 1
    #all right
    sort_psychos[4,1,:] = 1
    #right and no
    sort_psychos[5,1,3:] = 1
    sort_psychos[5,2,:3] = 1
    
    
    if psychos.shape[2]==8:
        del_ind = np.unique(np.where([psychos!=psychos])[-1])
        psychos = np.delete(psychos,del_ind,axis=2)
    if psychos.shape[2]==8:
        del_ind = [1,6]
        psychos = np.delete(psychos,del_ind,axis=2)

    similarity = compare_psychos(sort_psychos,psychos)
    state_similarity = np.nanargmin(similarity,axis=0)
    state_similarity = np.nanargmin(similarity,axis=1)

    while len(np.unique(state_similarity)) != len(state_similarity):
        states, counts = np.unique(state_similarity,return_counts=True)
        for confused_state in states[np.where(counts>1)[0]]:
            confused_ind = np.where(state_similarity == confused_state)[0]
            confused_state_similarity = similarity[confused_ind,:]
            replace_state_ind = np.argmax(confused_state_similarity[:,confused_state])
            replace_state_similarity = similarity[confused_ind[replace_state_ind],:]
            repalcement_state = np.argsort(replace_state_similarity)[1]
            state_similarity[confused_ind[replace_state_ind]]=repalcement_state
        print('fixed with ' + str(state_similarity))
        
        
    all_trials = pd.DataFrame()
    for i,trials in enumerate(hmm_trials):
        if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'post_hoc_pupil_diameter'
        elif all([len(trials.query('hmm_state==0'))>10,np.isin('pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'pupil_diameter'
       
    measures = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10']
    for measure in measures:
        all_trials=all_trials.query(measure+'=='+measure)

    face_z = scipy.stats.zscore(all_trials['face_energy'])
    run_z = scipy.stats.zscore(all_trials['running_speed'])
    all_trials['movement'] = face_z+run_z

    state_trials = all_trials.query('hmm_state==hmm_state')
    for state in np.unique(state_trials['hmm_state'].values):
        state_data = state_trials.query('hmm_state==@state')[pupil_measure].values
        state_data= state_data[state_data==state_data]
    good_pupil = state_trials.query(pupil_measure+'=='+pupil_measure)

    ########## movement
    bin_size=.25
    p_state=np.full([int(10/bin_size),6],np.nan)
    x_vals = np.arange(-5,5,bin_size)
    for k,p in enumerate(x_vals):
        p2 = p+bin_size
        p_trials = good_pupil.query('movement>=@p and movement<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,state_similarity[int(state[j])]]=counts[int(j)]/sum(counts)
                    state_probs_move[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)

      
    #
    k = np.where(state_probs_move[m,:,:]==state_probs_move[m,:,:])[1][0]
    k2 = np.where(state_probs_move[m,:,:]==state_probs_move[m,:,:])[1][-1]
    for state in np.unique(good_pupil['hmm_state']).astype(int):
        these_state_probs = state_probs_move[m,state_similarity[0],:]
        
    p2=x_vals[k+1]
    p=-7
    p_trials = good_pupil.query('movement>=@p and movement<@p2')
    state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
    for j in range(len(counts)):
        if sum(counts)>5:
            p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
            state_probs_move[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    state_probs_move[m,:,:k]=np.nan    
    
    for state in np.unique(good_pupil['hmm_state']):
        these_probs = state_probs_move[m,state_similarity[int(state)],k:k2+1]
        these_probs[these_probs!=these_probs]=0
        state_probs_move[m,state_similarity[int(state)],k:k2+1] = these_probs
  
    measures = ['post_hoc_pupil_diameter','face_energy','running_speed']

    for measure in measures:
        all_trials=all_trials.query(measure+'=='+measure)
    
    face = all_trials['face_energy'].to_numpy()
    face_z = stats.zscore(face)
    run = all_trials['running_speed'].to_numpy()
    run_z = stats.zscore(run)
    movement = face_z+run_z
    all_trials['movement'] = movement
    hmm_state = all_trials['hmm_state'].to_numpy()
    opt_ind = np.where(hmm_state==0)[0]
    sub_ind = np.where(hmm_state>1)[0]
    dis_ind = np.where(hmm_state==1)[0]
    indi_ind = np.where(hmm_state!=hmm_state)[0]
    pup = all_trials['post_hoc_pupil_diameter'].to_numpy()
    
    for i,data in enumerate([[face,run],[face,movement],[run,movement],[pup,face],[pup,run],[pup,movement]]):
        
        coeffs = np.polyfit(data[0],data[1],1)
        this_poly = np.poly1d(coeffs)
        
        r,p= stats.pearsonr(data[0],data[1])
        plt.title('r = '+str(r.round(3))+'\np = '+f"{p:.02g}")
        all_r[m,i] = r

    
    for i,data in enumerate([face,run,movement]):
        for s,ind in enumerate([opt_ind,dis_ind,sub_ind,indi_ind]):
            r,p= stats.pearsonr(pup[ind],data[ind])
            all_states_r[m,s,i]=r
    
m = 5 
hmm_trials_path = list(folder.glob('*hmm_trials*'))[m]
print(hmm_trials_path)
hmm_trials = np.load(hmm_trials_path, allow_pickle = True)

all_trials = pd.DataFrame()
for i,trials in enumerate(hmm_trials):
    if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
        all_trials = all_trials.append(trials)
        pupil_measure = 'post_hoc_pupil_diameter'
    elif all([len(trials.query('hmm_state==0'))>10,np.isin('pupil_diameter',trials.columns)]):
        all_trials = all_trials.append(trials)
        pupil_measure = 'pupil_diameter'

measures = ['post_hoc_pupil_diameter','face_energy','running_speed']

for measure in measures:
    all_trials=all_trials.query(measure+'=='+measure)

face = all_trials['face_energy'].to_numpy()
face_z = stats.zscore(face)
run = all_trials['running_speed'].to_numpy()
run_z = stats.zscore(run)
movement = face_z+run_z
all_trials['movement'] = movement
hmm_state = all_trials['hmm_state'].to_numpy()
opt_ind = np.where(hmm_state==0)[0]
sub_ind = np.where(hmm_state>1)[0]
dis_ind = np.where(hmm_state==1)[0]
indi_ind = np.where(hmm_state!=hmm_state)[0]

pup = all_trials['post_hoc_pupil_diameter'].to_numpy()

plt.figure(figsize=[16,20])

ax=[]
xlabels=['Face motion energy (a.u.)','Movement index (a.u.)','Movement index (a.u.)']
ylabels=['Locomotion speed (m/s)','Face motion energy (a.u.)','Locomotion speed (m/s)']
for i,data in enumerate([[face,run],[movement,face],[movement,run]]):
    ax.append(plt.subplot(6,4,i+1))

    plt.plot(data[0],data[1],'o',color=[.7,.7,.7],markersize=1)
    plt.xlabel(xlabels[i])
    plt.ylabel(ylabels[i])
    
    if np.isin(i,[0,2]):
        plt.yticks(np.arange(-.05,.36,.05),['','0.0','','0.1','','0.2','','0.3',''])
    if i==0:
        plt.xlim(0,1)
        plt.xticks(np.arange(0,1.1,.2))
    
    coeffs = np.polyfit(data[0],data[1],1)
    this_poly = np.poly1d(coeffs)
    
    if i > 0:
        plt.plot([data[0].min(),data[0].max()],[this_poly(data[0].min()),this_poly(data[0].max())],'k')
    
    r,p= stats.pearsonr(data[0],data[1])
    if i > 0:
        plt.title('r = '+str(r.round(3)))
        plt.xlim(-5,5)
    all_r[m,i] = r
    
ax.append(plt.subplot(6,4,4))
bp = plt.boxplot(all_r[:,2],patch_artist=True,showfliers=False,positions=[0])
for data in all_r[:,2]:
    plt.plot(np.random.randn(1)/20,data,'ko',zorder=10)
plt.ylim(0,1)
plt.xlim(-.25,.25)
plt.ylabel('movement index/measure\ncorrelation (r)')
plt.xticks([])



for patch in bp['boxes']:
    patch.set_facecolor([.7,.7,.7])
    patch.set_linewidth(0)
for line in bp['medians']:
    line.set_color('k')
    line.set_linewidth(3)
for line in bp['caps']:
    line.set_color('k')
    line.set_linewidth(2)
for line in bp['whiskers']:
    line.set_color('k')
    line.set_linewidth(2)

for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)

ax[0].text(-.2,.375,'A',fontsize=30)
ax[0].text(1.1,.375,'B',fontsize=30)
ax[0].text(3.75,.375,'C',fontsize=30)

bin_size=.25
p_state=np.full([int(10/bin_size),6],np.nan)
  
mouse_order = [1,5,0,7,8,2,9,10,3,11,12,4,6]

Tasks = ['Aud. task','Vis. task']
step = .02
bins_lower = np.arange(0,1,step)
for i, m in enumerate(mouse_order):
    sp_ind = int(i+7)
    if i==12:
        sp_ind=21

    plt.subplot(7,3,sp_ind)
    plt.title(subjects[m])
    if np.isin(sp_ind,[7,8]):
        plt.title('Auditory task\n\n' + subjects[m])
    if sp_ind== 9:
        plt.title('Visual task\n\n' + subjects[m])
 
    bin_size=.25
    mouse_data = state_probs_move[m,:,:40]
    x=np.arange(-5,5,bin_size)
    for state in range(6):
        y= mouse_data[state,:]
        plt.plot(x,y,color = cols[state])
    plt.xlim([-5,5])
    plt.ylim([0,1])
    plt.xticks(np.arange(-5,5.1,1),[])
    
    if np.isin(sp_ind,[16,17,21]):
        plt.xticks(np.arange(-5,5.1,1),['',-4,'',-2,'',0,'',2,'',4,''])
        plt.xlabel('Movement index (a.u.)')

    plt.yticks([0,.5,1],[])
    
    if np.isin(sp_ind,[7,10,13,16,21]):
        plt.yticks([0,.5,1],[0,.5,1])
        plt.ylabel('p(state)')


    