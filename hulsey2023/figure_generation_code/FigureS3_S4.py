import os
import numpy as np
import glob
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import sem
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

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]

base_folder = input("Enter the main directory path") + '\\'
hmm_paths = glob.glob(base_folder + 'hmms\\*hmm.npy')
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')


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



measures = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10']
subject_states = {}
subjects=[]
state_probs = np.full([len(hmm_trials_paths),6,50],np.nan)
state_probs_face = np.full([len(hmm_trials_paths),6,20],np.nan)
state_probs_run = np.full([len(hmm_trials_paths),6,40],np.nan)
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
        
    
    # classify GLM-HMM states
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
    # print(state_similarity)

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
        
    #drop nans and make movement index
    for measure in measures:
        all_trials=all_trials.query(measure+'=='+measure)

    face_z = scipy.stats.zscore(all_trials['face_energy'])
    run_z = scipy.stats.zscore(all_trials['running_speed'])
    all_trials['movement'] = face_z+run_z

    state_trials = all_trials.query('hmm_state==hmm_state')
    
    ### calculate p(state) over pupil bins
    bin_size=.02
    p_state=np.full([int(1/bin_size),6],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = state_trials.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,state_similarity[int(state[j])]]=counts[int(j)]/sum(counts)
                    state_probs[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    # grab trials that were not sufficient for their own bin
    k = np.where(state_probs[m,:,:]==state_probs[m,:,:])[1][0]
    k2 = np.where(state_probs[m,:,:]==state_probs[m,:,:])[1][-1]

    p2=(k+1)*bin_size
    p=0
    p_trials = state_trials.query(pupil_measure+'>=@p and '+pupil_measure+'<@p2')
    state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
    for j in range(len(counts)):
        if sum(counts)>5:
            p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
            state_probs[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    state_probs[m,:,:k]=np.nan
    
    # set nans to zero if data at other states
    for state in np.unique(state_trials['hmm_state']):
        these_probs = state_probs[m,state_similarity[int(state)],k:k2+1]
        these_probs[these_probs!=these_probs]=0
        state_probs[m,state_similarity[int(state)],k:k2+1] = these_probs
    
    
    ### face energy
    bin_size=.05
    p_state=np.full([int(1/bin_size),6],np.nan)
    for k,p in enumerate(np.arange(0,1,bin_size)):
        p2 = p+bin_size
        p_trials = state_trials.query('face_energy>=@p and face_energy<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,state_similarity[int(state[j])]]=counts[int(j)]/sum(counts)
                    state_probs_face[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    
    
    k = np.where(state_probs_face[m,:,:]==state_probs_face[m,:,:])[1][0]
    k2 = np.where(state_probs_face[m,:,:]==state_probs_face[m,:,:])[1][-1]
       
        
    p2=(k+1)*bin_size
    p=0
    p_trials = state_trials.query('face_energy>=@p and face_energy<@p2')
    state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
    for j in range(len(counts)):
        if sum(counts)>5:
            p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
            state_probs_face[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    state_probs_face[m,:,:k]=np.nan
    
    for state in np.unique(state_trials['hmm_state']):
        these_probs = state_probs_face[m,state_similarity[int(state)],k:k2+1]
        these_probs[these_probs!=these_probs]=0
        state_probs_face[m,state_similarity[int(state)],k:k2+1] = these_probs

    
    ######### running
    bin_size=.02
    p_state=np.full([50,6],np.nan)
    for k,p in enumerate(np.arange(-.02,.301,bin_size)):
        p2 = p+bin_size
        if p==.30:
            p2 = 1
        if p==-.02:
            p=-1
        p_trials = state_trials.query('running_speed>=@p and running_speed<@p2')
        if len(p_trials)>0:
            state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
            for j in range(len(counts)):
                if sum(counts)>5:
                    p_state[k,state_similarity[int(state[j])]]=counts[int(j)]/sum(counts)
                    state_probs_run[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)

    
    #
    k = np.where(state_probs_run[m,:,:]==state_probs_run[m,:,:])[1][0]
    k2 = np.where(state_probs_run[m,:,:]==state_probs_run[m,:,:])[1][-1]
        
    p2=(k+1)*bin_size
    p=0
    p_trials = state_trials.query('running_speed>=@p and running_speed<@p2')
    state,counts = np.unique(p_trials['hmm_state'],return_counts=True)
    for j in range(len(counts)):
        if sum(counts)>5:
            p_state[k,int(state[j])]=counts[int(j)]/sum(counts)
            state_probs_run[m,state_similarity[int(state[j])],k]=counts[int(j)]/sum(counts)
    state_probs_run[m,:,:k]=np.nan
    
    # state,counts = np.unique(good_pupil['hmm_state'],return_counts=True)
    for state in np.unique(state_trials['hmm_state']):
        these_probs = state_probs_run[m,state_similarity[int(state)],k:k2+1]
        these_probs[these_probs!=these_probs]=0
        state_probs_run[m,state_similarity[int(state)],k:k2+1] = these_probs

   


# load polyfitting selection of optimal pupil
modality = np.array(modality)
aud_ind = np.where(modality==0)[0]
vis_ind = np.where(modality==1)[0]

poly_selection_file = glob.glob(base_folder + 'misc\\pup*poly*degree*.npy')
poly_rsqs_file = glob.glob(base_folder + 'misc\\pup*poly*rsqs*.npy')

poly_degree = np.load(poly_selection_file[0])
rsqs = np.load(poly_rsqs_file[0])

mouse_order = [1,0,5,2,7,3,8,4,9,6,10,11,12]

Tasks = ['Aud. task','Vis. task']
optimal_pupils=np.full(len(hmm_trials_paths),np.nan)
step = .02
bins_lower = np.arange(0,1,step)

# plot individual mice, pupil polynomial fitting, and p(states)
fig = plt.figure(figsize=[18,20])
for i, m in enumerate(mouse_order):
    mouse_rsqs = rsqs[m,:,:]
    mouse_rsqs = np.mean(mouse_rsqs,axis=1)
    sp_ind = i*2+5

    if i==11:
        sp_ind = 29
    if i==12:
        sp_ind = 33
            
    
    # plot p(state) for each mouse, fit appropriate polynomial to optimal and plot
    plt.subplot(9,4,sp_ind+1)
    mouse_state_probs = state_probs[m,:,:]
    for i,state_prob in enumerate(mouse_state_probs):
        plt.plot(bins_lower,state_prob,color=cols[i],zorder=10-i)
        if i==0:
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
          
            plt.plot(opt_xs,ys,'k',zorder=10)
            plt.plot([this_opt_pupil,this_opt_pupil],[0,1],'k--',zorder=10)
            plt.xlim([0,1]) 
        
    plt.xlim([.25,1])
    plt.xticks(np.arange(.25,1.1,.125),['25','','50','','75','','100'])
    plt.yticks([0,.5,1])
    if sp_ind==23:
        plt.xlabel('Pupil diameter (%max)')
    elif sp_ind==33:
        plt.xlabel('Pupil diameter (%max)')
    else:
        plt.xticks(np.arange(.25,1.1,.125),[])
    plt.ylabel('p(state)')
    
    
    # plot r^2 values of test set polynomial fitting
    plt.subplot(9,4,sp_ind)
    plt.plot(np.arange(1,8),mouse_rsqs,'k')
    plt.plot(poly_degree[m],mouse_rsqs[poly_degree[m]-1],'sk')
    plt.xticks(np.arange(1,8,1),[1,'',3,'',5,'',7])
    if sp_ind==23:
        plt.xlabel('Polynomial degree')
    elif sp_ind==33:
        plt.xlabel('Polynomial degree')
    else:
        plt.xticks(np.arange(1,8,1),[])
    
    if min(mouse_rsqs)>.75:
        plt.ylim(.75,1)
        plt.yticks([.8,.9,1],['0.80','0.90','1.00'])
    elif min(mouse_rsqs)>.5:
        plt.ylim([.5,1])
        plt.yticks([.6,.8,1],['0.60','0.80','1.00'])

    else:
        plt.ylim([0,1])
        plt.yticks([0,.25,0.5,.75,1],['0.00','','0.50','','1.00'])

    plt.ylabel(subjects[m] + '\n' + 'Test set r$^2$')

plt.tight_layout()


# plot histogram of polynomial orders
plt.subplot(3,4,11)

plt.hist(poly_degree,width=.5,bins=100,color='k')
plt.xticks([2.25,3.25],[2,3])
plt.xlabel('Polynomial degree of best fit')
plt.ylabel('# Mice')
plt.yticks([0,4,8])
plt.xlim(1.5,4)
    
# plot r^2 of linear vs quadratic vs best fit
plt.subplot(3,4,12)

lineardata = rsqs[:,0,:].mean(axis=1)
data_2deg = rsqs[:,1,:].mean(axis=1)
all_rsqs = rsqs[:,:,:].mean(axis=2)
data=[]
for m,degree in enumerate(poly_degree):
    data.append(all_rsqs[m,degree-1])
plt.boxplot([lineardata,data_2deg,data],showfliers=False)
x = np.ones(len(data))+np.random.randn(len(data))*.05
plt.plot(x,lineardata,'ko')
x = 1+np.ones(len(data))+np.random.randn(len(data))*.05
plt.plot(x,data_2deg,'ko')
x = 2+np.ones(len(data))+np.random.randn(len(data))*.05
plt.plot(x,data,'ko')
plt.ylim([-.5,1])
plt.yticks(np.arange(-.5,1.1,.5))

plt.ylabel('Test set r$^2$')
plt.xticks([1,2,3],['Linear','Quadratic','Best fit'])

    
for spine in ['top','right']:
    for axs in fig.axes:
        axs.spines[spine].set_visible(False)

ax = plt.subplot(9,1,1)
plt.text(.2175,-.1,'Auditory task',ha='center',fontsize=20)
plt.text(.75,-.1,'Visual task',ha='center',fontsize=20)

plt.text(.225,.9,'Optimal\npupil diameter',ha='center',va='top')
plt.plot([.175,.175],[.6,.9],'k--')

plt.text(.325,.9,'Optimal',ha='center',va='top')
plt.plot([.29,.36],[.675,.675],color=cols[0],linewidth=3)

plt.text(.425,.9,'Bias left',ha='center',va='top')
plt.plot([.39,.46],[.675,.675],color=cols[2],linewidth=3)

plt.text(.525,.9,'Avoid right',ha='center',va='top')
plt.plot([.49,.56],[.675,.675],color=cols[3],linewidth=3)

plt.text(.625,.9,'Bias right',ha='center',va='top')
plt.plot([.59,.66],[.675,.675],color=cols[4],linewidth=3)

plt.text(.725,.9,'Avoid left',ha='center',va='top')
plt.plot([.69,.76],[.675,.675],color=cols[5],linewidth=3)

plt.text(.825,.9,'Disengaged',ha='center',va='top')
plt.plot([.79,.86],[.675,.675],color=cols[1],linewidth=3)


plt.xticks([])
plt.yticks([])


plt.text(.1,.9,'Polynomial fit',ha='center',va='top')
y= np.array([.575,.7,.575])
x2= np.array([.065,.1,.135])
opt_xs = np.arange(x2[0],x2[-1],.001)
this_poly_degree = 2

coeffs = np.polyfit(x2.astype(np.float32),y.astype(np.float32),this_poly_degree)
this_poly = np.poly1d(coeffs)

ys = this_poly(opt_xs)
# this_opt_pupil = opt_xs[ys.argmax()].round(5)
# optimal_pupils[m]=this_opt_pupil
  
plt.plot(opt_xs,ys,'k',zorder=10,clip_on=False)
plt.xlim([0,1]) 



for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_visible(False)


plt.ylim([0,1])
plt.xlim([0,1])
# plt.savefig(base_folder + 'figures\\figure_S2_v2.pdf')
# plt.savefig(base_folder + 'figures\\figure_S2_v2.png', dpi=300)

# ############# p(state) by Raw movement measures of individual mice

mouse_order = [1,0,5,2,7,3,8,4,9,6,10,11,12]

Tasks = ['Aud. task','Vis. task']
step = .02
bins_lower = np.arange(0,1,step)
fig = plt.figure(figsize=[18,20])
for i, m in enumerate(mouse_order):
    sp_ind = int(i+np.floor(i/2)*2+1)

    if i==11:
        sp_ind = 25
    if i==12:
        sp_ind = 29
            
    
    plt.subplot(8,4,sp_ind)
    bin_size=.05
    mouse_data = state_probs_face[m,:,:]

    x=np.arange(0,1,bin_size)
    for state in range(6):
        y= mouse_data[state,:]
        plt.plot(x,y,color = cols[state])
        plt.xlim([0,1])
        plt.ylim([0,1])
        
    if sp_ind==19:
        plt.xlabel('Face motion energy (%max)')
        plt.xticks(np.arange(0,1.1,.25))
    elif sp_ind==29:
        plt.xlabel('Face motion energy (%max)')
        plt.xticks(np.arange(0,1.1,.25))
    else:
        plt.xticks(np.arange(0,1.1,.25),[])
        
    if sp_ind%4==1:
        plt.ylabel('p(state)')
        plt.yticks([0,.5,1])

    else:
        plt.yticks([0,.5,1],[])

    plt.title(subjects[m])

    if i==0:
        plt.title('Auditory task\n\n'+subjects[m])
    elif i==1:
        plt.title('Visual task\n\n'+subjects[m])
        
        
    plt.subplot(8,4,sp_ind+2)
    bin_size=.02
    mouse_data = state_probs_run[m,:,:]
    # nan_ind = np.where(np.nansum(mouse_data,axis=0)==0)[0]
    # mouse_data[mouse_data!=mouse_data]=0
    # mouse_data[:,nan_ind]=np.nan
    bin_size=.02
    basex=np.arange(-.02,.301,bin_size)
    for state in range(6):
        y= mouse_data[state,:]
        plt.plot(basex,y[:len(basex)],color = cols[state])
        plt.xlim([-.03,.3])    
        plt.ylim([0,1])
   
    if sp_ind==19:
        plt.xlabel('Locomotion speed (m/s)')
        plt.xticks(np.arange(0,.31,.1),['0.0','0.1','0.2','0.3+'])
    elif sp_ind==29:
        plt.xlabel('Locomotion speed (m/s)')
        plt.xticks(np.arange(0,.31,.1),['0.0','0.1','0.2','0.3+'])
    else:
        plt.xticks(np.arange(0,.31,.1),[])
    plt.yticks([0,.5,1],[])

    plt.title(subjects[m])
    if i==0:
        plt.title('Auditory task\n\n'+subjects[m])
    elif i==1:
        plt.title('Visual task\n\n'+subjects[m])
            
        
plt.subplot(8,4,2)
plt.ylabel('\np(state)')
plt.subplot(8,4,3)
plt.ylabel('\np(state)')
plt.subplot(8,4,4)
plt.ylabel('\np(state)')
plt.suptitle('        Face motion energy                                                                                        Locomotion speed\n')

plt.tight_layout()

plt.subplot(8,4,2)
plt.ylabel('')
plt.subplot(8,4,3)
plt.ylabel('')
plt.subplot(8,4,4)
plt.ylabel('')


# average face energy
ax= plt.subplot(3,4,10)
l=[]
alpha = .2
basex= np.arange(0,1,.05)
state_data=[]
for state in range(3):
    mode_data = state_probs_face[:,state,:]
    if state==2:
        all_mouse_sub_opt=[]
        for mouse_data in state_probs_face:
            sub_opt_data=[]
            states,bins = np.where(mouse_data==mouse_data)
            states = np.unique(states)[2:]
            for this_state in states:
                sub_opt_data.append(mouse_data[this_state,:])
                
            
            mouse_sub_opt= np.full(state_probs_face.shape[2],np.nan)
            mouse_sub_opt[np.unique(bins)] = np.nansum(np.vstack(sub_opt_data)[:,np.unique(bins)],axis=0)
            all_mouse_sub_opt.append(mouse_sub_opt)
            
        mode_data = np.vstack(all_mouse_sub_opt)
             
    mean=[]
    lower=[]
    upper=[]
    err=[]
    for qt in range(mode_data.shape[1]):
        bin_data = mode_data[:,qt]
        bin_data = bin_data[bin_data==bin_data]
        mean.append(np.mean(bin_data))
        err.append(sem(bin_data))
    mean = np.array(mean)
    err = np.array(err)
    state_data.append(mean)

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
    l.append(plt.plot(basex[keep_ind],mean,'k',linewidth=4,color=cols[state])[0])
    ax.add_patch(p)


plt.xlabel('Face motion energy (% max)')
plt.legend(l,['optimal',
              'disengaged',
              'suboptimal'])
plt.xlim([0,.8])
plt.ylim([0,1])
tick = np.arange(0,.81,.20)
label = np.arange(0,81,20).astype(str)
plt.xticks(np.arange(0,1.1,.25))

tick = np.arange(0,1.1,.2)
label = tick.round(1).astype(str)
plt.yticks(tick,label)

plt.title('All subjects')


plt.yticks(np.arange(0,1.1,.2))
plt.ylabel('p(state)')


#average locomotion speed
        
l=[]
ax = plt.subplot(3,4,12)
alpha = .2
# basex= np.arange(-1,1,.05)
bin_size=.02
basex=np.arange(-.02,.301,bin_size)
state_data=[]
for state in range(3):
    mode_data = state_probs_run[:,state,:]
    if state==2:
        all_mouse_sub_opt=[]
        for mouse_data in state_probs_run:
            sub_opt_data=[]
            states,bins = np.where(mouse_data==mouse_data)
            states = np.unique(states)[2:]
            for this_state in states:
                sub_opt_data.append(mouse_data[this_state,:])
                
            
            mouse_sub_opt= np.full(state_probs_run.shape[2],np.nan)
            mouse_sub_opt[np.unique(bins)] = np.nansum(np.vstack(sub_opt_data)[:,np.unique(bins)],axis=0)
            all_mouse_sub_opt.append(mouse_sub_opt)
            
        mode_data = np.vstack(all_mouse_sub_opt)
        
    mean=[]
    lower=[]
    upper=[]
    err=[]
    for qt in range(mode_data.shape[1]):
        bin_data = mode_data[:,qt]
        bin_data = bin_data[bin_data==bin_data]
        mean.append(np.mean(bin_data))
        err.append(sem(bin_data))
    mean = np.array(mean)
    err = np.array(err)
    state_data.append(mean)

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
    l.append(plt.plot(basex[keep_ind],mean,'k',linewidth=4,color=cols[state])[0])
    ax.add_patch(p)


plt.xlabel('Locomotion speed (m/s)')
plt.legend(l,['optimal',
              'disengaged',
              'suboptimal'])
plt.ylim([0,1])
tick = np.arange(0,.31,.1)
label = tick.round(1).astype(str)
label[-1] = label[-1] + '+'
plt.xticks(tick,label)
plt.xlim([-.025,.3])

tick = np.arange(0,1.1,.2)
label = tick.round(1).astype(str)
label[:]=''
plt.yticks(tick,label)

plt.title('All subjects')


for spine in ['top','right']:
    for axs in fig.axes:
        axs.spines[spine].set_visible(False)


# plt.savefig(base_folder + 'figures\\figure_S3_v2.pdf')
# plt.savefig(base_folder + 'figures\\figure_S3_v2.png', dpi=300)
#######################
