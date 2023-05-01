import os
import numpy as np
import glob
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import sem
from uobrainflex.nwb import loadbehavior as load
from uobrainflex.behavioranalysis import flex_hmm
import scipy
from matplotlib.patches import Rectangle

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f,'k','k','k','k','k','k','k','k']

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt

#get saved hmm_trials paths
base_folder = 'D:\\Hulsey\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

ax=[]
fig = plt.figure(figsize=[16,12])


#### Panel A, image of mouse face
ax.append(plt.subplot(position=[.025,.7,.3,.275]))

face_file = base_folder +'images\\mouse_face_annotated.png'
face_image_data = plt.imread(face_file)

ax[-1].imshow(face_image_data)
ax[-1].axis('off')

##### Panel A2, arousal measures traces
#enter index for desired subject and load data
m=6
subject = os.path.basename(hmm_trials_paths[m])[:5]
hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
dwell_data = flex_hmm.get_dwell_times(hmm_trials)

# enter index for desried session and laod raw data 
session = 3
nwb_path = base_folder + '\\nwb\\' + subject + '\\' + hmm_trials[session]['file_name'].iloc[0]

behavior_measures = load.load_behavior_measures(nwb_path)
trial_data = hmm_trials[session]
session_dwell = dwell_data[session]

trial_range = [428,450]
range_start_time = round(trial_data['start_time'].iloc[trial_range[0]]*1000-4000)
range_end_time = round(trial_data['start_time'].iloc[trial_range[1]]*1000-13000)


ax.append(plt.subplot(position=[.425,.775,.55,.2]))
st = trial_data['start_time'].values[trial_range[0]:trial_range[1]]*1000

ax[-1].plot(behavior_measures['post_hoc_pupil_diameter'].iloc[range_start_time:range_end_time])
pupil = trial_data['post_hoc_pupil_diameter'].values
ax[-1].scatter(st,pupil[trial_range[0]:trial_range[1]],c='k', zorder=10)
ax[-1].text(range_start_time-2500,.75,'Pupil\ndiameter'
            ,color=[.12, .46, .7],ha='right',va='center')

ax[-1].plot(behavior_measures['face_energy'].iloc[range_start_time:range_end_time]/1.5+.025)
face = trial_data['face_energy'].values
ax[-1].scatter(st,face[trial_range[0]:trial_range[1]]/1.5+.025,c='k',marker='s', zorder=10)
ax[-1].text(range_start_time-2500,.35,'Face motion\nenergy'
            ,color=[1, .5, .05],ha='right',va='center')


ax[-1].plot(behavior_measures['running_speed'].iloc[range_start_time:range_end_time]/3)
run = trial_data['running_speed'].values
ax[-1].scatter(st,run[trial_range[0]:trial_range[1]]/3,c='k',marker='>', zorder=10)
ax[-1].text(range_start_time-2500,-.1,'Locomotion\nspeed'
            ,color=[.17, .63, .17],ha='right',va='bottom')

for i, trial in enumerate(st):
    ax[-1].add_patch(Rectangle((round(trial),0), round(1.2*1000), 1,color='k',alpha=.2,edgecolor = None,linewidth=0))
    
if len(session_dwell.index)>0:
    for idx in session_dwell.index: 
        start_trial = session_dwell.loc[idx,'first_trial']
        end_trial = start_trial + session_dwell.loc[idx,'dwell']-1
        start_time = trial_data['start_time'].iloc[start_trial]-.5
        duration = trial_data['start_time'].iloc[end_trial] - start_time +1
        ax[-1].add_patch(Rectangle((round(start_time*1000),0), round(duration*1000), 1
                                   ,color=cols[int(session_dwell.loc[idx,'state'])],alpha=.4
                                   ,edgecolor = None,linewidth=0))

ax[-1].set_xticks([])
ax[-1].set_xlim([trial_data['start_time'].iloc[trial_range[0]]*1000-4000,
               trial_data['start_time'].iloc[trial_range[1]]*1000-13000])
ax[-1].set_ylim([-.05,1])
ax[-1].axis('off')


legend_center = range_start_time-17500
ax[-1].add_patch(Rectangle((legend_center-13500,-.325), 27000, .125
                            ,color=[.8,.8,.8],alpha=.4,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center,-.275,'HMM states'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center-13500,-.4875), 27000, .125
                            ,color=cols[0],alpha=.4,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center,-.4375,'Optimal'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center+14500,-.4875), 27000, .125
                            ,color=cols[1],alpha=.4,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center+28000,-.4375,'Disengaged'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center+42500,-.4875), 27000, .125
                            ,color=cols[2],alpha=.4,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center+56000,-.4375,'Suboptimal'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((range_start_time+4000,-.275), 1200, .275
                            ,color='k',alpha=.2,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(range_start_time+5500,-.275,'Stimulus'
            ,color='k',ha='left',va='bottom')

ax[-1].plot([range_start_time,3857600],[0,-.57],'k',clip_on=False)
ax[-1].plot([range_end_time,3868200],[0,-.57],'k',clip_on=False)


# Panel B, full session trial measure values
ax.append(plt.subplot(position = [0.125, .375, 0.85, .3]))
ax.append(plt.subplot(position = [0.125, .38, 0.85, .295]))

st = trial_data['start_time'].values*1000
ax[-1].scatter(st,pupil-.1,c='k',s=10, zorder=10,clip_on=False)
ax[-1].scatter(st,face/2+.07,c='k',marker='s',s=10, zorder=10,clip_on=False)
ax[-1].scatter(st,run/3,c='k',marker='>',s=10, zorder=10,clip_on=False)
if len(session_dwell.index)>0:
    for idx in session_dwell.index: 
        start_trial = session_dwell.loc[idx,'first_trial']
        end_trial = start_trial + session_dwell.loc[idx,'dwell']-1
        
        start_time = trial_data['start_time'].iloc[start_trial] - .5
        duration = trial_data['start_time'].iloc[end_trial] - start_time + 1
        ax[-1].add_patch(Rectangle((round(start_time*1000),-.01), round(duration*1000), .885,
           color=cols[int(session_dwell.loc[idx,'state'])],alpha=.4,clip_on=False
           ,edgecolor = None,linewidth=0))
        
ax[-1].plot([0,0],[0,.2/3],'k',linewidth=3,clip_on=False)
ax[-1].text(-75000,.033,'Loc. speed\n20 cm/s',ha='right',va='center')

ax[-1].plot([0,0],[.25,.25+.2/2],'k',linewidth=3,clip_on=False)
ax[-1].text(-75000,.3,'Face m.e.\n20% max',ha='right',va='center')

ax[-1].plot([0,0],[.5,.7],'k',linewidth=3,clip_on=False)
ax[-1].text(-75000,.6,'Pupil\ndiameter\n20% max',ha='right',va='center')


ax[-1].plot([360000,660000],[.75,.75],'k',linewidth=3)
ax[-1].text(510000,.775,'5 minutes',ha='center',va='bottom')
ax[-1].axis('off')
ax[-1].set_xlim([trial_data['start_time'].iloc[0]*1000,trial_data['start_time'].iloc[-1]*1000])
ax[-1].set_ylim([-.025,.875])



###########
measures = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10']
subject_states = {}
subjects=[]
norm_pupil_hist=np.full([len(hmm_trials_paths),6,50],np.nan)
state_probs = np.full([len(hmm_trials_paths),3,50],np.nan)
state_probs_move = np.full([len(hmm_trials_paths),3,100],np.nan)
modality=[]
for m in range(len(hmm_trials_paths)):
    subject = os.path.basename(hmm_trials_paths[m])[:5]
    subjects.append(subject)
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
    modality.append(hmm_trials[0]['target_modality'].iloc[0])
    trial_date=[]
    for trials in hmm_trials:
        trial_date.append(trials['file_name'].iloc[0])
    
    hmm_trials=hmm_trials[np.argsort(trial_date)]        
    
    all_trials = pd.DataFrame()
    for i,trials in enumerate(hmm_trials):
        if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'post_hoc_pupil_diameter'
        elif all([len(trials.query('hmm_state==0'))>10,np.isin('pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'pupil_diameter'

        
        
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
    good_pupil['hmm_state'].iloc[np.where(good_pupil['hmm_state']>2)[0]]=2
    bin_size=.02
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = good_pupil.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    state_probs[m,int(state[j]),k]=counts[int(j)]/sum(counts)
    k=np.where(p_state[:,0]==p_state[:,0])[0][0]
    p2=(k+1)*bin_size
    p=0
    p_trials = good_pupil.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
    if len(p_trials)>0:
        state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
        for j in range(len(counts)):
            if sum(counts)>5:
                p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                state_probs[m,int(state[j]),k]=counts[int(j)]/sum(counts)
       
    bin_size=.25
    p_state=np.full([int(10/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(-5,5,bin_size)):
        p2 = p+bin_size
        p_trials = good_pupil.query('movement>=@p and movement<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    state_probs_move[m,int(state[j]),k]=counts[int(j)]/sum(counts)
 

# load polyfitting selection of optimal pupil
modality = np.array(modality)
aud_ind = np.where(modality==0)[0]
vis_ind = np.where(modality==1)[0]

poly_selection_file = glob.glob(base_folder + 'misc\\*poly*degree*.npy')
poly_degree = np.load(poly_selection_file[0])


Tasks = ['Aud. task','Vis. task']
optimal_pupils=np.full(len(hmm_trials_paths),np.nan)
step = .02
bins_lower = np.arange(0,1,step)
for m in range(len(modality)):
    mouse_state_probs = state_probs[m,:,:]
    state_prob = mouse_state_probs[0,:]
    y= state_prob
    x2=bins_lower[y==y]
    y=y[y==y]    
    opt_xs = np.arange(x2[0],x2[-1],.001)
    this_poly_degree = poly_degree[m]
            
    coeffs = np.polyfit(x2.astype(np.float32),y.astype(np.float32),this_poly_degree)
    this_poly = np.poly1d(coeffs)
    
    ys = this_poly(opt_xs)
    this_opt_pupil = opt_xs[ys.argmax()].round(5)
    optimal_pupils[m]=this_opt_pupil
    
x = np.arange(0,1,.02)
optimal_shift = np.full(state_probs.shape,np.nan) 
for m, mouse_pupil in enumerate(optimal_pupils):
    optimal_bin=np.argmin(abs(x-mouse_pupil))
    for j,this_bin in enumerate(np.arange(optimal_bin-25,optimal_bin+25)):
        if this_bin<state_probs.shape[2]:
            if j<50:
                optimal_shift[m,:,j]=state_probs[m,:,this_bin]

# Panel C, example polyfit
i=6 #example mouse ID -- BW058

ax.append(plt.subplot(position=[.075,.05,.125,.275]))

bin_size=.02
mouse_data = state_probs[i,:,:]
nan_ind = np.where(np.nansum(mouse_data,axis=0)==0)[0]
mouse_data[mouse_data!=mouse_data]=0
mouse_data[:,nan_ind]=np.nan
x=np.arange(0,100,bin_size*100)
for state in [2,1,0]:
    y= mouse_data[state,:]
    x2=x[y==y]
    y=y[y==y]

    ax[-1].plot(x2,y,color=cols[state],linewidth=2)
    ax[-1].plot(x2,y,'o',color=cols[state])
    
    if state==0:
        ax[-1].plot(x2,y,color=cols[state],linewidth=4)
        ax[-1].plot(x2,y,'o',color=cols[state],markersize=10)
        coeffs = np.polyfit(x2,y,2)
        poly = np.poly1d(coeffs)    
        ax[-1].plot(x,poly(x),'k--',linewidth=1)
        ax[-1].set_ylim([0,1])
        
        optimal_pupil = -coeffs[1]/(coeffs[0]*2)
ax[-1].set_xlim([50,100])
ticks = np.arange(50,101,10)
labels = ticks
ax[-1].set_xticks(ticks)
ax[-1].set_xticklabels(labels)
ax[-1].set_xlabel('Pupil diameter (%max)')
ax[-1].set_yticks(np.arange(0,1.1,.2))
ax[-1].set_ylabel('p(state)')

ax[-1].arrow(optimal_pupil,.95,0,-.1, width=.5, color="k"
      ,head_width=6, head_length=.05, overhang=.1,zorder=12)
ax[-1].text(80,.9,'N = 1\nmouse',ha='left',va='top')
ax[-1].text(optimal_pupil,.95,'Optimal\npupil diameter',ha='center',va='bottom')
######## Panel D, aud vs vis optimal pupil

ax.append(plt.subplot(position=[.265,.05,.1,.275]))

optimal_pupils = np.array(optimal_pupils) # first axis: median, poly2, poly3, poly4

aud = optimal_pupils[np.where(np.array(modality)==0)[0]]
vis = optimal_pupils[np.where(np.array(modality)==1)[0]]
data = [aud*100,vis*100]

bp = ax[-1].boxplot(data,showfliers = False,positions=[0,1],widths=.25,patch_artist=True)
for i in range(2):
    ax[-1].plot(np.ones(len(data[i]))*i+np.random.randn(len(data[i]))/20,data[i],'ko',zorder=10)
    
ax[-1].set_ylim([33.5,82.5])
ax[-1].set_ylabel('Optimal pupil diameter (%max)')
ax[-1].set_xticks([0,1])
ax[-1].set_xticklabels(['Auditory','Visual'])
ax[-1].set_xlim([-.25,1.25])


for patch in bp['boxes']:
    patch.set_facecolor([.5,.5,.5])
    patch.set_linewidth(0)
for line in bp['medians']:
    line.set_color('k')
    line.set_linewidth(2)
for line in bp['caps']:
    line.set_color('k')
    line.set_linewidth(2)
for line in bp['whiskers']:
    line.set_color('k')
    line.set_linewidth(2)   

ax[-1].plot([0,1],[83,83],'k',clip_on=False,linewidth=2)
ax[-1].text(.5,84,'* *',ha='center',va='bottom')

###################### Panel E, delta opt pupil vs state probs
l=[]
ax.append(plt.subplot(position=[.415,.05,.125,.275]))

alpha = .2
basex= np.arange(-50,50,2)

for state in range(3):
    state_data = optimal_shift[:,state,:]

    mean=[]
    lower=[]
    upper=[]
    err=[]
    for qt in range(state_data.shape[1]):
        bin_data = state_data[:,qt]
        bin_data = bin_data[bin_data==bin_data]
        mean.append(np.mean(bin_data))
        err.append(sem(bin_data))
    mean = np.array(mean)
    err = np.array(err)
    
    keep_ind = np.where(mean==mean)[0]
    mean = mean[keep_ind]
    err = err[keep_ind]
    err = np.array(err)
    err[np.where(err!=err)[0]] = 0
    upper =  np.add(mean,err)
    lower = np.subtract(mean,err)
    y=  np.concatenate((upper, np.flip(lower)))
    x= basex[keep_ind]
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor=cols[state],alpha=alpha)
    l.append(ax[-1].plot(basex[keep_ind],mean,'k',linewidth=4,color=cols[state])[0])
    ax[-1].add_patch(p)


ax[-1].set_yticks(np.arange(0,1.1,.2))
ax[-1].set_ylabel('p(state)')
ax[-1].set_xlabel('\u0394 Optimal pupil (% max)')
# ax[-1].legend(l,['Optimal',
#               'Disengaged',
#               'Sub-optimal'])
tick = np.arange(-20,21,10)
label = tick.astype(str)
ax[-1].set_xticks(tick)
ax[-1].set_xticklabels(label)

tick = np.arange(0,1.1,.2)
label = tick.round(1).astype(str)
ax[-1].set_yticks(tick)
ax[-1].set_yticklabels(label)

ax[-1].set_xlim([-25,25])
ax[-1].set_ylim([0,1])
ax[-1].set_title('Pupil diameter')
ax[-1].text(5,.9,'N = 13',ha='left',va='top')

############# Panel F, movement index vs state probs

l=[]
ax.append(plt.subplot(position=[.5875,.05,.125,.275]))
alpha = .2
basex= np.arange(-5,5,.25)
bin_size=.25
basex=np.arange(-5,5,bin_size)
for state in range(3):
    state_data = state_probs_move[:,state,:]

    mean=[]
    lower=[]
    upper=[]
    err=[]
    for qt in range(state_data.shape[1]):
        bin_data = state_data[:,qt]
        bin_data = bin_data[bin_data==bin_data]
        mean.append(np.mean(bin_data))
        err.append(sem(bin_data))
    mean = np.array(mean)
    err = np.array(err)
    
    keep_ind = np.where(mean==mean)[0]
    mean = mean[keep_ind]
    err = err[keep_ind]
    err = np.array(err)
    err[np.where(err!=err)[0]] = 0
    upper =  np.add(mean,err)
    lower = np.subtract(mean,err)
    y=  np.concatenate((upper, np.flip(lower)))
    x= basex[keep_ind]
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor=cols[state],alpha=alpha)
    l.append(ax[-1].plot(basex[keep_ind],mean,'k',linewidth=4,color=cols[state])[0])
    ax[-1].add_patch(p)


ax[-1].set_xlabel('Movement (z-score)')
ax[-1].set_ylim([0,1])
ax[-1].set_xlim([-4,4])
tick = np.arange(-4,4.1,2)
label = tick.round(0).astype(int).astype(str)
ax[-1].set_xticks(tick)
ax[-1].set_xticklabels(label)

tick = np.arange(0,1.1,.2)
label = tick.round(1).astype(str)
label[:]=''
ax[-1].set_yticks(tick)
ax[-1].set_yticklabels(label)

ax[-1].set_title('Movement index')
ax[-1].text(1,.9,'N = 13',ha='left',va='top')



#### Panel G, enter exit opt measure variability

all_mouse_state_enter_means_no_opt=[]
all_mouse_state_exit_means_no_opt=[]

mouse_n=[]
modality=[]
subjects=[]
for m in range(len(hmm_trials_paths)):
    hmm_tirals_path = hmm_trials_paths[m]
    hmm_trials = np.load(hmm_tirals_path,allow_pickle=True)
    dwell_data = flex_hmm.get_dwell_times(hmm_trials)
    subject = os.path.basename(hmm_tirals_path)[:5]
    subjects.append(subject)
    modality.append(hmm_trials[0]['target_modality'].iloc[0])    
    
    
    measures_to_use = ['post_hoc_pupil_diameter', 'post_hoc_pupil_diff', 'post_hoc_pupil_std10',
                       'face_energy', 'face_energy', 'face_std_10',
                       'running_speed','running_speed_diff', 'running_std10']
    
    
    enter_means=[]
    exit_means=[]
    no_opt_enter_means=[]
    no_opt_exit_means=[]
    no_opt_enter_means_all=[]
    no_opt_exit_means_all=[]
    dis_enter_means=[]
    dis_exit_means=[]
    sub_enter_means=[]
    sub_exit_means=[]
    opt_enter_means=[]
    opt_exit_means=[]
    for sp, measure in enumerate(measures_to_use):
        n=0
        measure_transition_enter_matrix = np.full([5000,20],np.nan)  
        measure_transition_exit_matrix = np.full([5000,20],np.nan) 
        prior_state= np.full(5000,np.nan)
        next_state= np.full(5000,np.nan)
        dwell= np.full(5000,np.nan)
        for sess in range(len(hmm_trials)):
            session_trials = hmm_trials[sess]
            session_dwells = dwell_data[sess]
            state_ind = np.where(session_dwells['state']==0)[0]
            for SI in state_ind:
                state_end = session_dwells['last_trial'].iloc[SI]
                state_start = session_dwells['first_trial'].iloc[SI]
                
                end_data = session_trials[measure].iloc[state_end-10:state_end+10].values
                start_data = session_trials[measure].iloc[state_start-10:state_start+10].values
    
                if len(start_data) == 20:
                    measure_transition_enter_matrix[n,:]=start_data
                if len(end_data) == 20:
                    measure_transition_exit_matrix[n,:]=end_data
                
                if SI>0:
                    prior_state[n]=session_dwells['state'].iloc[SI-1]
                if SI<len(session_dwells)-1:
                    next_state[n]=session_dwells['state'].iloc[SI+1]
                dwell[n]=session_dwells['dwell'].iloc[SI]
                n=n+1
        
            
        enter_data = measure_transition_enter_matrix[:n,:]
        exit_data = measure_transition_exit_matrix[:n,:]
        
        
        good_ind=np.where(dwell>9)[0]

        ind = good_ind[np.isin(good_ind,np.where(prior_state>0)[0])]
        no_opt_enter_means.append(np.nanmean(enter_data[ind,:],axis=0))
        ind = good_ind[np.isin(good_ind,np.where(next_state>0)[0])]        
        no_opt_exit_means.append(np.nanmean(exit_data[ind,:],axis=0))
 
    all_mouse_state_enter_means_no_opt.append(np.vstack(no_opt_enter_means))
    all_mouse_state_exit_means_no_opt.append(np.vstack(no_opt_exit_means))
    mouse_n.append(n)
    
#only opt epochs where exit or enter to another state
enter_data = np.dstack(all_mouse_state_enter_means_no_opt).swapaxes(1,2)
exit_data = np.dstack(all_mouse_state_exit_means_no_opt).swapaxes(1,2)


ax.append(plt.subplot(position=[.825,.0125,.15,.35]))   
ind=[2,5,8]
l=[]
for j, i in enumerate(ind):
    this_enter = enter_data[i,:,:]
    this_exit = exit_data[i,:,:]
    
    this_enter = this_enter - this_enter[:,:].mean(axis=1)[:,None] 
    this_exit = this_exit - this_exit[:,:].mean(axis=1)[:,None]     
    
    enter_mean = 1-j*.01+this_enter.mean(axis=0)
    enter_error = sem(this_enter,axis=0)
    exit_mean = 1-j*.01+this_exit.mean(axis=0)
    exit_error = sem(this_exit,axis=0)
    
    y = 1-j*.01
    ax[-1].plot([0,42],[y,y],'k--')
    
    upper =  np.add(enter_mean,enter_error)
    lower = np.subtract(enter_mean,enter_error)
    y=  np.concatenate((upper, np.flip(lower)))
    x= np.arange(20)
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor='k',alpha=.2)        
    ax[-1].add_patch(p)
    l.append(ax[-1].plot(x[:20],enter_mean,'k',linewidth=2)[0])

    upper =  np.add(exit_mean,exit_error)
    lower = np.subtract(exit_mean,exit_error)
    y=  np.concatenate((upper, np.flip(lower)))
    x= np.arange(22,42)
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor='k',alpha=.2)        
    ax[-1].add_patch(p)
    ax[-1].plot(x[:20],exit_mean,'k',linewidth=2)

ax[-1].add_patch(Rectangle((9.5,.97), 22, .04,color=cols[0],alpha=.4,edgecolor = None,linewidth=0))
 
ax[-1].set_xticks([])
ax[-1].set_xlim([-1,42])
ax[-1].set_ylim([.972,1.005])
ax[-1].set_yticks([.98,.99,1])
ax[-1].set_yticklabels(['Loc.\nspeed','Face\nenergy','Pupil\ndiameter'])
ax[-1].set_ylabel('Δ standard deviation')

ax[-1].plot([2,2],[.9935,.996],'k',clip_on=False)
ax[-1].text(1,.9935,'0.25%',ha='right',va='bottom')

ax[-1].plot([2,2],[.974,.9765],'k',clip_on=False)
ax[-1].text(1,.974,'0.025 cm/s',ha='right',va='bottom')

ax[-1].plot([16,26],[1.003,1.003],'k',linewidth=2)
ax[-1].text(21,1.0048,'10 trials',ha='center',va='top')
ax[-1].spines['left'].set_visible(False)
ax[-1].spines['bottom'].set_visible(False)
ax[-1].tick_params(axis=u'both', which=u'both',length=0)


for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)
        
        
        
ax[0].text(-75,50,'a',fontsize=30,weight="bold")
ax[0].text(-75,625,'b',fontsize=30,weight="bold")
ax[4].text(28.5,1.05,'c',fontsize=30,weight="bold")
ax[4].text(105,1.05,'d',fontsize=30,weight="bold")
ax[4].text(167.5,1.05,'e',fontsize=30,weight="bold")
ax[4].text(245,1.05,'f',fontsize=30,weight="bold")
ax[4].text(315,1.05,'g',fontsize=30,weight="bold")
        

plt.savefig(base_folder + 'figures\\figure_3.pdf')
plt.savefig(base_folder + 'figures\\figure_3.png', dpi=300)
