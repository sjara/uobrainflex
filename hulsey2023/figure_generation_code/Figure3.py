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
from scipy import stats 

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
base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

ax=[]
fig = plt.figure(figsize=[16,12])

#### Panel A, image of mouse face
ax.append(plt.subplot(position=[.025,.7,.3,.275]))

face_file = base_folder +'images\\mouse_face_annotated.png'
face_image_data = plt.imread(face_file)

ax[-1].imshow(face_image_data)
ax[-1].axis('off')

##### Panel A2, arousal measures traces of example session
#enter index for desired subject and load data
m=6
subject = os.path.basename(hmm_trials_paths[m])[:5]
hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
dwell_data = flex_hmm.get_dwell_times(hmm_trials)

# enter index for desried session and load raw data 
session = 3
nwb_path = base_folder + '\\nwb\\' + subject + '\\' + hmm_trials[session]['file_name'].iloc[0]

behavior_measures = load.load_behavior_measures(nwb_path)
trial_data = hmm_trials[session]
session_dwell = dwell_data[session]

trial_range = [428,450]
range_start_time = round(trial_data['start_time'].iloc[trial_range[0]]*1000-4000)
range_end_time = round(trial_data['start_time'].iloc[trial_range[1]]*1000-13000)

# plot behavior measures data
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

# plot sitmuli
alpha=.2
for i, trial in enumerate(st):
    ax[-1].add_patch(Rectangle((round(trial),0), round(1.2*1000), 1,color='k',alpha=alpha,edgecolor = None,linewidth=0))
    
#plot hmm states
if len(session_dwell.index)>0:
    for idx in session_dwell.index: 
        start_trial = session_dwell.loc[idx,'first_trial']
        end_trial = start_trial + session_dwell.loc[idx,'dwell']-1
        start_time = trial_data['start_time'].iloc[start_trial]-.5
        duration = trial_data['start_time'].iloc[end_trial] - start_time +1
        ax[-1].add_patch(Rectangle((round(start_time*1000),0), round(duration*1000), 1
                                   ,color=cols[int(session_dwell.loc[idx,'state'])],alpha=alpha
                                   ,edgecolor = None,linewidth=0))

ax[-1].set_xticks([])
ax[-1].set_xlim([trial_data['start_time'].iloc[trial_range[0]]*1000-4000,
               trial_data['start_time'].iloc[trial_range[1]]*1000-13000])
ax[-1].set_ylim([-.05,1])
ax[-1].axis('off')


# make legend
legend_center = range_start_time-17500
ax[-1].add_patch(Rectangle((legend_center-13500,-.325), 27000, .125
                            ,color=[.8,.8,.8],alpha=alpha,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center,-.275,'HMM states'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center-13500,-.4875), 27000, .125
                            ,color=cols[0],alpha=alpha,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center,-.4375,'Optimal'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center+14500,-.4875), 27000, .125
                            ,color=cols[1],alpha=alpha,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center+28000,-.4375,'Disengaged'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((legend_center+42500,-.4875), 27000, .125
                            ,color=cols[2],alpha=alpha,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(legend_center+56000,-.4375,'Suboptimal'
            ,color='k',ha='center',va='center')

ax[-1].add_patch(Rectangle((range_start_time+4000,-.275), 1200, .275
                            ,color='k',alpha=alpha,clip_on=False
                            ,edgecolor = None,linewidth=0))
ax[-1].text(range_start_time+5500,-.275,'Stimulus'
            ,color='k',ha='left',va='bottom')

ax[-1].plot([range_start_time,3857600],[0,-.57],'k',clip_on=False)
ax[-1].plot([range_end_time,3868200],[0,-.57],'k',clip_on=False)


# Panel B, full session trial measure values
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
           color=cols[int(session_dwell.loc[idx,'state'])],alpha=alpha,clip_on=False
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



########### Gather p(state) vs measures data
measures = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10']
subject_states = {}
subjects=[]
state_probs = np.full([len(hmm_trials_paths),3,50],np.nan)
state_probs_move = np.full([len(hmm_trials_paths),3,100],np.nan)

move_loc_corr=[]
move_face_corr=[]
modality=[]

for m in range(len(hmm_trials_paths)):
    subject = os.path.basename(hmm_trials_paths[m])[:5]
    subjects.append(subject)
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)
    modality.append(hmm_trials[0]['target_modality'].iloc[0])
 
    all_trials = pd.DataFrame()
    for i,trials in enumerate(hmm_trials):
        if all([len(trials.query('hmm_state==0'))>10,np.isin('post_hoc_pupil_diameter',trials.columns)]):
            all_trials = all_trials.append(trials)
            pupil_measure = 'post_hoc_pupil_diameter'

    for measure in measures:
        all_trials=all_trials.query(measure+'=='+measure)
    
    #calculate movement index
    face = all_trials['face_energy']
    run = all_trials['running_speed']
    face_z = scipy.stats.zscore(face)
    run_z = scipy.stats.zscore(run)
    move = face_z+run_z
    all_trials['movement'] = face_z+run_z
        
    # get movement x face or run correlations
    r,p= stats.pearsonr(move,run)
    move_loc_corr.append(r)
    r,p= stats.pearsonr(move,face)
    move_face_corr.append(r)

    # take only trials in discrete state
    state_trials = all_trials.query('hmm_state==hmm_state')
    state_trials['hmm_state'].iloc[np.where(state_trials['hmm_state']>2)[0]]=2
    
    # calculate p(state) per pupil bin
    bin_size=.02
    p_state=np.full([int(1/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = state_trials.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    state_probs[m,int(state[j]),k]=counts[int(j)]/sum(counts)
    
    # get not included edge data and add to adjascent bin
    k=np.where(p_state[:,0]==p_state[:,0])[0][0]
    p2=(k+1)*bin_size
    p=0
    p_trials = state_trials.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
    if len(p_trials)>0:
        state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
        for j in range(len(counts)):
            if sum(counts)>5:
                p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                state_probs[m,int(state[j]),k]=counts[int(j)]/sum(counts)
       
    # calculate p(state) per move index bin
    bin_size=.25
    p_state=np.full([int(10/bin_size),3],np.nan)
    for k,p in enumerate(np.arange(-5,5,bin_size)):
        p2 = p+bin_size
        p_trials = state_trials.query('movement>=@p and movement<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
                    state_probs_move[m,int(state[j]),k]=counts[int(j)]/sum(counts)
 

# load coss validate polynomial degree selection of for optimal pupil
poly_selection_file = glob.glob(base_folder + 'misc\\pupil*poly*degree*.npy')
poly_degree = np.load(poly_selection_file[0])
Tasks = ['Aud. task','Vis. task']
optimal_pupils=np.full(len(hmm_trials_paths),np.nan)
step = .02
bins_lower = np.arange(0,1,step)
for m in range(len(hmm_trials_paths)):
    # get optimal state probability by pupil diameter per subject
    opt_prob = state_probs[m,0,:]
    y= opt_prob
    x2=bins_lower[y==y]
    y=y[y==y]    
    opt_xs = np.arange(x2[0],x2[-1],.001)
    
    #fit polynomial to p(opt) data
    this_poly_degree = poly_degree[m]
    coeffs = np.polyfit(x2.astype(np.float32),y.astype(np.float32),this_poly_degree)
    this_poly = np.poly1d(coeffs)
    
    ys = this_poly(opt_xs)
    this_opt_pupil = opt_xs[ys.argmax()].round(5)
    # save optimal pupil diameter for each subject
    optimal_pupils[m]=this_opt_pupil
    
    
# step through all mice and shift their p(state) curves to center optimal pupil
x = np.arange(0,1,.02)
optimal_shift = np.full(state_probs.shape,np.nan) 
for m, mouse_pupil in enumerate(optimal_pupils):
    optimal_bin=np.argmin(abs(x-mouse_pupil))
    for j,this_bin in enumerate(np.arange(optimal_bin-25,optimal_bin+25)):
        if this_bin<state_probs.shape[2]:
            if all([this_bin<50, this_bin>=0]):
                optimal_shift[m,:,j]=state_probs[m,:,this_bin]

# Panel C, example p(opt) and optimal pupil polyfit
i=6 #example mouse ID -- BW058

ax.append(plt.subplot(position=[.075,.05,.2,.275]))

bin_size=.02
mouse_data = state_probs[i,:,:]
# make sure that any pupils diameters without data are nans instead of zeros
nan_ind = np.where(np.nansum(mouse_data,axis=0)==0)[0]
mouse_data[mouse_data!=mouse_data]=0
mouse_data[:,nan_ind]=np.nan
x=np.arange(0,100,bin_size*100)

#plot example mouse adta
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
      ,head_width=3, head_length=.05, overhang=.1,zorder=12)
ax[-1].text(80,.9,'N = 1\nmouse',ha='left',va='top')
ax[-1].text(optimal_pupil,.95,'Optimal\npupil diameter',ha='center',va='bottom')
######## Panel D, aud vs vis optimal pupil

ax.append(plt.subplot(position=[.35,.05,.1,.275]))

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
ax[-1].set_xlabel('Task modality')
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

###################### Panel E, p(state) by delta opt pupil
l=[]
ax.append(plt.subplot(position=[.525,.05,.2,.275]))

alpha = .2
basex= np.arange(-50,50,2)

for state in range(3):
    state_data = optimal_shift[:,state,:]

    mean=[]
    lower=[]
    upper=[]
    err=[]
    # iterate across bins and get stats across mice with data available
    for qt in range(state_data.shape[1]):
        bin_data = state_data[:,qt]
        bin_data = bin_data[bin_data==bin_data]
        mean.append(np.mean(bin_data))
        err.append(sem(bin_data))
    mean = np.array(mean)
    err = np.array(err)
    
    #discard nans
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
ax[-1].set_xlabel('\u0394 Optimal pupil (% max)')
# ax[-1].legend(l,['Optimal',
#               'Disengaged',
#               'Sub-optimal'])
tick = np.arange(-20,21,10)
label = tick.astype(str)
ax[-1].set_xticks(tick)
ax[-1].set_xticklabels(label)

ax[-1].set_ylabel('p(state)')
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
ax.append(plt.subplot(position=[.775,.05,.2,.275]))
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


ax[-1].set_xlabel('Movement index (a.u.)')
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

ax[-1].set_title('Movement')
ax[-1].text(1,.9,'N = 13',ha='left',va='top')

tick = np.arange(0,1.1,.2)
ax[-1].set_yticks(tick)
ax[-1].set_yticklabels([])


for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)
        

ax[0].text(-75,50,'A',fontsize=30)
ax[0].text(-75,625,'B',fontsize=30)
ax[4].text(37,1.05,'C',fontsize=30)
ax[4].text(105,1.05,'D',fontsize=30)
ax[4].text(150,1.05,'E',fontsize=30)
ax[4].text(220,1.05,'F',fontsize=30)
        

# plt.savefig(base_folder + 'figures\\figure_3.pdf')
# plt.savefig(base_folder + 'figures\\figure_3.png', dpi=300)
