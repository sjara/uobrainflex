import os
import numpy as np
import glob
import pandas as pd
from scipy import stats
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
from pathlib import Path
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,[.6,.6,.6]]

base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')
hmm_paths = glob.glob(base_folder + 'hmms\\*hmm.npy')

### gather mean values of measures per mouse


mean_measure = np.full([len(hmm_trials_paths),6,3],np.nan) #[mouse,state,measure]

subjects=[]
all_r = np.full([len(hmm_trials_paths),4],np.nan)
all_mouse_state_enter_means_no_opt=[]
all_mouse_state_exit_means_no_opt=[]

for m in range(len(hmm_trials_paths)):
    print(str(m+1) + ' of ' + str(len(hmm_paths)))
    subject = os.path.basename(hmm_paths[m])[:5]
    subjects.append(subject)
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle=True)

    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        all_trials = all_trials.append(trials)
        
    measures_to_use = ['post_hoc_pupil_std10','face_std_10','running_std10'] 
    for measure in np.concatenate([measures_to_use,['post_hoc_pupil_diameter']]):
        all_trials=all_trials.query(measure+'=='+measure) 
        
    face_z = stats.zscore(all_trials['face_energy'])
    run_z = stats.zscore(all_trials['running_speed'])
    movement = face_z+run_z
    all_trials['movement'] = movement 
           
    for zz, measure in enumerate(measures_to_use):
        
        for state in np.unique(all_trials['hmm_state'].dropna()):
            state_trials = all_trials.query('hmm_state==@state')
            data = state_trials[measure].dropna().values
            mean_measure[m,int(state),zz]=data.mean()
        state_trials = all_trials.query('hmm_state>1')

        data = state_trials[measure].dropna().values
        mean_measure[m,-1,zz]=data.mean() ### merged sub optimal tacked on at end

    ## gather movement vs pupil correlations per state
    
    for state in [0,1]:
        state_trials = all_trials.query('hmm_state==@state')
        mov = state_trials['movement']
        pup = state_trials['post_hoc_pupil_diameter']
        r = round(np.corrcoef(mov,pup)[0,1],3)        
        all_r[m,state] = r
        
    state_trials = all_trials.query('hmm_state>1')
    mov = state_trials['movement']
    pup = state_trials['post_hoc_pupil_diameter']
    r = round(np.corrcoef(mov,pup)[0,1],3)        
    all_r[m,2] = r
    
    state_trials = all_trials.query('hmm_state!=hmm_state')
    mov = state_trials['movement']
    pup = state_trials['post_hoc_pupil_diameter']
    r = round(np.corrcoef(mov,pup)[0,1],3)        
    all_r[m,3] = r
        
    ## gather enter/exit opt state data
    dwell_data = flex_hmm.get_dwell_times(hmm_trials)    
    no_opt_enter_means=[]
    no_opt_exit_means=[]
    for sp, measure in enumerate(measures_to_use): # for each measure
        n=0
        measure_transition_enter_matrix = np.full([5000,20],np.nan)     ##[instance,data]
        measure_transition_exit_matrix = np.full([5000,20],np.nan)      ##[instance,data]
        prior_state= np.full(5000,np.nan)
        next_state= np.full(5000,np.nan)
        dwell= np.full(5000,np.nan)
        for sess in range(len(hmm_trials)):
            session_trials = hmm_trials[sess]
            session_dwells = dwell_data[sess]
            opt_state_ind = np.where(session_dwells['state']==0)[0]         #look at optimal state instances
            for SI in opt_state_ind:
                state_end = session_dwells['last_trial'].iloc[SI]
                state_start = session_dwells['first_trial'].iloc[SI]
                
                # get measure values entering and exiting state
                end_data = session_trials[measure].iloc[state_end-10:state_end+10].values       
                start_data = session_trials[measure].iloc[state_start-10:state_start+10].values 
    
                #save if there is full data
                if len(start_data) == 20:
                    measure_transition_enter_matrix[n,:]=start_data
                if len(end_data) == 20:
                    measure_transition_exit_matrix[n,:]=end_data
                
                #record prior and next discrete state
                if SI>0:                                                    
                    prior_state[n]=session_dwells['state'].iloc[SI-1]
                if SI<len(session_dwells)-1:
                    next_state[n]=session_dwells['state'].iloc[SI+1]
                #record dwell time in state
                dwell[n]=session_dwells['dwell'].iloc[SI]
                n=n+1
        
        
        enter_data = measure_transition_enter_matrix[:n,:]
        exit_data = measure_transition_exit_matrix[:n,:]
        
        #use only states when dwell was >10 trials
        good_ind=np.where(dwell>9)[0]

        #use only transitions into or out of non optimal states
        ind = good_ind[np.isin(good_ind,np.where(prior_state>0)[0])]
        no_opt_enter_means.append(np.nanmean(enter_data[ind,:],axis=0))
        ind = good_ind[np.isin(good_ind,np.where(next_state>0)[0])]        
        no_opt_exit_means.append(np.nanmean(exit_data[ind,:],axis=0))
    
    all_mouse_state_enter_means_no_opt.append(np.vstack(no_opt_enter_means))
    all_mouse_state_exit_means_no_opt.append(np.vstack(no_opt_exit_means))
    
#only opt epochs where exit or enter to another state
opt_enter_data = np.dstack(all_mouse_state_enter_means_no_opt).swapaxes(1,2)
opt_exit_data = np.dstack(all_mouse_state_exit_means_no_opt).swapaxes(1,2)


#################### PLot figure


#plot 10 trial standard deviation for each measure by state
lims=[[0,.12],[0,.15],[0,.15]]
sbplt=[1,2,3]
ylbl = ['Pupil diameter\n10 trial std. (% max)',
        'Face m.e.\n10 trial std. (a.u.)','Locomotion speed\n10 trial std. (m/s)']
plt.figure(figsize=[16,8])
bplot=[]
ax=[]
for i in range(3):
    ax.append(plt.subplot(2,4,sbplt[i]))
    data = [mean_measure[:,1,i],mean_measure[:,0,i],mean_measure[:,2,i]]
    # print(i)
    # print('dis vs opt')
    # print(stats.wilcoxon(data[0],data[1]))
    # print('\nsub vs opt')
    # print(stats.wilcoxon(data[1],data[2]))
    # print('\nsub vs dis')
    # print(stats.wilcoxon(data[0],data[2]))
    bplot.append(plt.boxplot(data,positions=[0,1,2],showfliers=False,patch_artist=True))

    data = np.vstack(data)
    for this_mouse in data.T:
        x=np.arange(3)+np.random.randn(3)*.1
        plt.plot(x,this_mouse,color=[.8,.8,.8],zorder=10,marker='o')
    plt.ylim(lims[i])
        
    if i==0:
        plt.yticks(np.arange(0,.11,.025),np.arange(0,11,2.5))
    elif i==1:
        plt.yticks(np.arange(0,.16,.05),np.arange(0,16,5))
    elif i==2:
        plt.yticks(np.arange(0,.16,.05))
    
    plt.xticks([0,1,2],['Dis','Opt','Sub'])
    plt.ylabel(ylbl[i])

ax[0].set_title('Pupil diameter')
ax[1].set_title('Face motion energy')
ax[2].set_title('Locomotion speed')

ax[0].plot([.05,.95],[.10,.10],'k')
ax[0].text(.5,.10,'* * *',ha='center',va='bottom')
ax[0].plot([1.05,1.95],[.10,.10],'k')
ax[0].text(1.5,.10,'* *',ha='center',va='bottom')
ax[0].plot([0.05,1.95],[.11,.11],'k',clip_on=False)
ax[0].text(1,.11,'* * *',ha='center',va='bottom')

ax[1].plot([.05,.95],[.14,.14],'k')
ax[1].text(.5,.14,'* *',ha='center',va='bottom')
ax[1].plot([1.05,1.95],[.14,.14],'k')
ax[1].text(1.5,.14,'* *',ha='center',va='bottom')

ax[2].plot([.05,.95],[.125,.125],'k')
ax[2].text(.5,.125,'* *',ha='center',va='bottom')
ax[2].plot([.05,1.95],[.1375,.1375],'k')
ax[2].text(1,.1375,'*',ha='center',va='bottom')
ax[2].plot([1.05,1.95],[.125,.125],'k')
ax[2].text(1.5,.125,'*',ha='center',va='bottom')

# color box plots
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
        line.set_linewidth(2)
        line.set_zorder(11)
    for line in b['caps']:
        line.set_color('k')
        line.set_linewidth(1)
        line.set_zorder(11)
    for line in b['whiskers']:
        line.set_color('k')
        line.set_linewidth(1)   
        line.set_zorder(11)
        
####### plot enter/exit opt state
alpha = .2
ax.append(plt.subplot(2,4,4))   
l=[]
for i in range(3):
    this_enter = np.array(opt_enter_data[i,:,:])
    this_exit = np.array(opt_exit_data[i,:,:])
    
    #take the change in values to plot analagous to pairwise comparison
    this_enter = this_enter - this_enter[:,:].mean(axis=1)[:,None]  
    this_exit = this_exit - this_exit[:,:].mean(axis=1)[:,None]     
    
    # add offset to plot on same axis
    offset = .0135
    if i==2:
        offset = .016
    
    enter_mean = 1-i*offset+this_enter.mean(axis=0)
    enter_error = stats.sem(this_enter,axis=0)
    exit_mean = 1-i*offset+this_exit.mean(axis=0)
    exit_error = stats.sem(this_exit,axis=0)
    
    #plot line of no change
    y = 1-i*offset
    ax[-1].plot([0,42],[y,y],'k--')
    
    upper =  np.add(enter_mean,enter_error)
    lower = np.subtract(enter_mean,enter_error)
    y=  np.concatenate((upper, np.flip(lower)))
    x= np.arange(20)
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor='k',alpha=alpha)        
    ax[-1].add_patch(p)
    l.append(ax[-1].plot(x[:20],enter_mean,'k',linewidth=1,marker='o',markersize=3)[0])
    ax[-1].plot(x[9],enter_mean[9],'ko',markersize=7)
    ax[-1].plot(x[19],enter_mean[19],'ko',markersize=7)

    upper =  np.add(exit_mean,exit_error)
    lower = np.subtract(exit_mean,exit_error)
    y=  np.concatenate((upper, np.flip(lower)))
    x= np.arange(22,42)
    x = np. concatenate([x,np.flip(x)])
    p = Polygon(list(zip(x,y)),facecolor='k',alpha=alpha)        
    ax[-1].add_patch(p)
    ax[-1].plot(x[:20],exit_mean,'k',linewidth=1,marker='o',markersize=3)
    ax[-1].plot(x[9],exit_mean[9],'ko',markersize=7)
    ax[-1].plot(x[19],exit_mean[19],'ko',markersize=7)
    
    # print(i)
    # print('\nenter comparison')
    # print(stats.wilcoxon(this_enter[:,9],this_enter[:,19]))
    # print('\nexit comparison')
    # print(stats.wilcoxon(this_exit[:,9],this_exit[:,19]))
    
ax[-1].add_patch(Rectangle((9.5,.96), 22, .05,color=cols[0],alpha=alpha,edgecolor = None,linewidth=0))
 
ax[-1].set_xticks([])
ax[-1].set_xlim([-1,42])
ax[-1].set_ylim([.96,1.005])
ax[-1].set_yticks([.968,.9865,1])
ax[-1].set_yticklabels(['Loc.\nspeed','Face\nenergy','Pupil\ndiameter'])
ax[-1].set_ylabel('Δ standard deviation')

ax[-1].plot([2,2],[.991,.9935],'k',clip_on=False)
ax[-1].text(1,.99075,'0.25%',ha='right',va='bottom')

ax[-1].plot([2,2],[.9735,.976],'k',clip_on=False)
ax[-1].text(1,.97325,'0.25 cm/s',ha='right',va='bottom')

ax[-1].plot([0,9],[.965,.965],'k',linewidth=2,clip_on=False)
ax[-1].text(4.5,.9635,'10 trials',ha='center',va='top')
ax[-1].spines['left'].set_visible(False)
ax[-1].spines['bottom'].set_visible(False)
ax[-1].tick_params(axis=u'both', which=u'both',length=0)

ax[-1].plot([9,19],[1.004,1.004],'k',clip_on=False)
ax[-1].plot([9,9],[1.004,1.0037],'k',clip_on=False)
ax[-1].plot([19,19],[1.004,1.0037],'k',clip_on=False)
ax[-1].text(14,1.005,'p = .07',ha='center', fontsize=13)
ax[-1].plot([31,41],[1.004,1.004],'k',clip_on=False)
ax[-1].plot([41,41],[1.004,1.0037],'k',clip_on=False)
ax[-1].plot([31,31],[1.004,1.0037],'k',clip_on=False)
ax[-1].text(36,1.005,'* *',ha='center')

ax[-1].plot([9,19],[.99325,.99325],'k',clip_on=False)
ax[-1].plot([9,9],[.99295,.99325],'k',clip_on=False)
ax[-1].plot([19,19],[.99295,.99325],'k',clip_on=False)
ax[-1].text(14,.99375,'* *',ha='center')
ax[-1].plot([31,41],[.99325,.99325],'k',clip_on=False)
ax[-1].plot([31,31],[.99295,.99325],'k',clip_on=False)
ax[-1].plot([41,41],[.99295,.99325],'k',clip_on=False)
ax[-1].text(36,.99375,'p = .12',ha='center', fontsize=13)

ax[-1].plot([9,19],[.9755,.9755],'k',clip_on=False)
ax[-1].plot([9,9],[.9752,.9755],'k',clip_on=False)
ax[-1].plot([19,19],[.9752,.9755],'k',clip_on=False)
ax[-1].text(14,.976,'*',ha='center')
ax[-1].plot([31,41],[.9755,.9755],'k',clip_on=False)
ax[-1].plot([31,31],[.9752,.9755],'k',clip_on=False)
ax[-1].plot([41,41],[.9752,.9755],'k',clip_on=False)
ax[-1].text(36,.976,'* *',ha='center')


####### plot example pupil vs movement correlations per state
m = 6
hmm_trials_path = hmm_trials_paths[m]
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

all_pupil = all_trials['post_hoc_pupil_diameter'].to_numpy()
face_z = stats.zscore(all_trials['face_energy'])
run_z = stats.zscore(all_trials['running_speed'])
movement = face_z+run_z
all_trials['movement'] = movement
hmm_state = all_trials['hmm_state'].to_numpy()
opt_ind = np.where(hmm_state==0)[0]
sub_ind = np.where(hmm_state>1)[0]
dis_ind = np.where(hmm_state==1)[0]
indi_ind = np.where(hmm_state!=hmm_state)[0]


h=[]
l=[]
states=['Disengaged','Optimal','Suboptimal','Indeterminate']
mouse_r=[]
for z,ind in enumerate([dis_ind,opt_ind,sub_ind]):#,indi_ind]):
    ax.append(plt.subplot(2,4,5+z))
    mov = movement[ind]
    pup = all_pupil[ind]   
    plt.plot(mov,pup,'o',color=colors[z],markersize=1)
    
    coeffs = np.polyfit(mov,pup,1)
    this_poly = np.poly1d(coeffs)
    plt.plot([min(movement),max(movement)],[this_poly(min(mov)),this_poly(max(mov))],'k')


    plt.xlabel('Movement index (a.u.)')
    
    plt.ylim(.45,1)
    plt.xlim(-5,5)
    
    if z==0:
        plt.ylabel('Pupil diameter (% max)')
        plt.yticks(np.arange(.5,1.01,.1))
    else:
        plt.yticks(np.arange(.5,1.01,.1),'')

    plt.title(states[z])
    r = round(np.corrcoef(movement[ind],all_trials['post_hoc_pupil_diameter'].to_numpy()[ind])[0,1],3)   
    plt.text(-4,.925,'r = ' + str(r))
    # plt.title('r = ' + str(r))

subject = trials['file_name'].iloc[0][:5]


###### plot group data of pupil vs movement correlations
ax.append(plt.subplot(2,4,8))
b = plt.boxplot([all_r[:,1],all_r[:,0],all_r[:,2]],showfliers = False, patch_artist=True)
for i,data in enumerate(np.vstack([all_r[:,1],all_r[:,0],all_r[:,2]]).T):
    x = np.random.randn(len(data))/10+np.arange(1,4)
    if i == m:
        plt.plot(x,data,'k',marker='o',zorder=10)
    else:
        plt.plot(x,data,color=[.8,.8,.8],marker='o',zorder=10)
    
plt.ylabel('Pupil/movement correlation (r)')    
plt.xticks(np.arange(1,4),['Dis','Opt','Sub'])    
plt.xlabel('Performance state')

plt.plot([1.05,1.95],[.7,.7],'k',clip_on=False)
plt.plot([2.05,2.95],[.7,.7],'k',clip_on=False)
plt.plot([1.05,2.95],[.775,.775],'k',clip_on=False)
plt.text(1.5,.7,'* * *',ha='center')
plt.text(2.5,.7,'* *',ha='center')
plt.text(2,.775,'*',ha='center')
plt.ylim(-.22,.72)




for patch, color in zip(b['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_linewidth(0)
for line, color in zip(b['medians'], ['k','k','k','k']):
    line.set_color(color)
    line.set_linewidth(2)
    line.set_zorder(11)
for line in b['caps']:
    line.set_color('k')
    line.set_linewidth(1)
    line.set_zorder(11)
for line in b['whiskers']:
    line.set_color('k')
    line.set_linewidth(1)   
    line.set_zorder(11)


for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)
    
plt.tight_layout()

ax[0].text(-1.5,.12,'A',fontsize=30)
ax[0].text(11.75,.12,'B',fontsize=30)
ax[0].text(-1.5,-.025,'C',fontsize=30)
ax[0].text(11.75,-.025,'D',fontsize=30)


# plt.savefig(base_folder + 'figures\\figure_4.pdf')
# plt.savefig(base_folder + 'figures\\figure_4.png', dpi=300)
