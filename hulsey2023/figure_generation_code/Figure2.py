import os
import numpy as np
import glob
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import sem
from uobrainflex.behavioranalysis import flex_hmm
from matplotlib.patches import Rectangle

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt


def pdur(val,dur):
     return (1-val)*val**(dur-1)    
 
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


state_names = ['optimal','disengaged','left bias','avoid right','right bias','avoid left']

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

all_state_params  = np.full([len(hmm_trials_paths),6,2,2],np.nan) 
all_state_psychos = np.full([len(hmm_trials_paths),6,3,6],np.nan)

lh = np.full([len(hmm_trials_paths),6],np.nan) # left hit
le = np.full([len(hmm_trials_paths),6],np.nan) #left error
lm = np.full([len(hmm_trials_paths),6],np.nan) # left miss
rh = np.full([len(hmm_trials_paths),6],np.nan) #right hit
re = np.full([len(hmm_trials_paths),6],np.nan) #right error
rm = np.full([len(hmm_trials_paths),6],np.nan) #right miss
all_state_occupancy=[]
all_state_similarity=[]
state_stay_prob=np.full([len(hmm_trials_paths),6],np.nan)
ys = np.full([len(hmm_trials_paths),6,21],np.nan)
median_dwell = np.full([len(hmm_trials_paths),6],np.nan)
subjects=[]
dwell_histograms=np.full([len(hmm_trials_paths),4,100],np.nan)
dwell_CDFs=np.full([len(hmm_trials_paths),4,500],np.nan)

sort_psychos = np.zeros([6,3,6])
#optimal
sort_psychos[0,0,:3] = 1
sort_psychos[0,1,3:] = 1
#disengaged
sort_psychos[1,2,:] = 1
#bias left
sort_psychos[2,0,:] = 1
#avoid right
sort_psychos[3,0,:3] = 1
sort_psychos[3,2,3:] = 1
#bias right
sort_psychos[4,1,:] = 1
#aboid left
sort_psychos[5,1,3:] = 1
sort_psychos[5,2,:3] = 1
    
for m, mouse_path in enumerate(hmm_trials_paths):
    print(mouse_path)
    subject = os.path.basename(mouse_path)[:5]    
    subjects.append(subject)
    hmm_trials = np.load(mouse_path,allow_pickle=True)
    
    this_hmm = np.load(hmm_paths[m],allow_pickle= True)
    hmm = this_hmm.tolist()
    trans_mat = np.exp(hmm.transitions.log_Ps)

    #### get master psycho for each state and assign it a state number by comparing to standard
    
    all_trials = pd.DataFrame()
    new_hmm_trials = []
    for trials in hmm_trials:
        if np.unique(trials['hmm_state'])[0]==0: # only sessions with optimal state
            all_trials = all_trials.append(trials)
            new_hmm_trials.append(trials)
    hmm_trials = new_hmm_trials
        
    psychos, xvals = get_psychos(hmm_trials)
        
    if psychos.shape[2]==8: # some mice had 4 difficulties per side for a couple of sessions. drop these trials for state type classificaiton
        del_ind = np.unique(np.where([psychos!=psychos])[-1])
        psychos = np.delete(psychos,del_ind,axis=2)
    if psychos.shape[2]==8:
        del_ind = [1,6]
        psychos = np.delete(psychos,del_ind,axis=2)

    similarity = compare_psychos(sort_psychos,psychos)
    state_similarity = np.nanargmin(similarity,axis=1)

    while len(np.unique(state_similarity)) != len(state_similarity): # if any stats are not immediately classified, comapre more
        states, counts = np.unique(state_similarity,return_counts=True)
        for confused_state in states[np.where(counts>1)[0]]:
            confused_ind = np.where(state_similarity == confused_state)[0]
            confused_state_similarity = similarity[confused_ind,:]
            replace_state_ind = np.argmax(confused_state_similarity[:,confused_state])
            replace_state_similarity = similarity[confused_ind[replace_state_ind],:]
            repalcement_state = np.argsort(replace_state_similarity)[1]
            state_similarity[confused_ind[replace_state_ind]]=repalcement_state
        print('fixed with ' + str(state_similarity))
    # now we have state_similarity variable that indicates which type of state each state is per mouse
    all_state_similarity.append(state_similarity)
    
    
    for state in range(6):
        if len(all_trials.query('hmm_state==@state'))>0:
            if len(all_trials.query('hmm_state==@state'))>20:             
                lft_h = len(all_trials.query('hmm_state==@state and inpt<0 and choice==0'))/len(all_trials.query('hmm_state==@state and inpt<0'))
                lft_e = len(all_trials.query('hmm_state==@state and inpt<0 and choice==1'))/len(all_trials.query('hmm_state==@state and inpt<0'))
                lft_m = len(all_trials.query('hmm_state==@state and inpt<0 and choice==2'))/len(all_trials.query('hmm_state==@state and inpt<0'))
                rt_h = len(all_trials.query('hmm_state==@state and inpt>0 and choice==1'))/len(all_trials.query('hmm_state==@state and inpt>0'))
                rt_e = len(all_trials.query('hmm_state==@state and inpt>0 and choice==0'))/len(all_trials.query('hmm_state==@state and inpt>0'))
                rt_m = len(all_trials.query('hmm_state==@state and inpt>0 and choice==2'))/len(all_trials.query('hmm_state==@state and inpt>0'))
                lh[m,state_similarity[state]]=lft_h
                le[m,state_similarity[state]]=lft_e
                lm[m,state_similarity[state]]=lft_m
                rh[m,state_similarity[state]]=rt_h
                re[m,state_similarity[state]]=rt_e
                rm[m,state_similarity[state]]=rt_m
            
    occupancy=[]
    occupancy.append(len(np.where(all_trials['hmm_state']==0)[0])/len(all_trials))
    occupancy.append(len(np.where(all_trials['hmm_state']==1)[0])/len(all_trials))
    occupancy.append(len(np.where(all_trials['hmm_state']>1)[0])/len(all_trials))
    occupancy.append(len(np.where(all_trials['hmm_state']!=all_trials['hmm_state'])[0])/len(all_trials))

    all_state_occupancy.append(np.array(occupancy))
    
    # get dwell time data
    for ii, state in enumerate(state_similarity):
        state_stay_prob[m,state] = trans_mat[ii,ii]
    dwell_data = flex_hmm.plot_dwell_times(subject, hmm_trials,0)
    mouse_dwells=pd.DataFrame()
    all_dwells=pd.DataFrame()
    ind_dwell=[]
    
    # get dwell times in each state. compare time between discrete state start/ends to get discrete state transition times (indetermine stated wells)
    for this_data in dwell_data:
        session_ind_dwell = this_data['first_trial'].to_numpy() - np.append(np.array([0]),this_data['last_trial'].to_numpy()[:-1])
        ind_dwell.append(session_ind_dwell)
        if this_data.iloc[-1]['state']==1:
            if this_data.iloc[-1]['dwell']>40:
                mouse_dwells = mouse_dwells.append(this_data[:-1])
            else:
                mouse_dwells = mouse_dwells.append(this_data)
        else:
            mouse_dwells = mouse_dwells.append(this_data)
        all_dwells = all_dwells.append(this_data)
    
    ind_dwell = np.concatenate(ind_dwell)
         
    
    # make cumulative distribution function for dwell time data
    for state in range(3):
        data = mouse_dwells.query('state==@state')['dwell'].values
        if state==2:
            data = mouse_dwells.query('state>=@state')['dwell'].values
        
        for i in range(500):
            dwell_CDFs[m,state,i] = len(np.where(data<=i)[0])/len(data)
        y,x = np.histogram(data,bins=np.arange(0,501,5))
        dwell_histograms[m,state,:]=y
    
    for i in range(500):
        dwell_CDFs[m,3,i] = len(np.where(ind_dwell<=i)[0])/len(ind_dwell)
    y,x = np.histogram(ind_dwell,bins=np.arange(0,501,5))
    dwell_histograms[m,3,:]=y
     
    # get median dwell time data
    for state in np.unique(all_dwells['state']):
        data = all_dwells.query('state==@state')['dwell'].values
        print(np.median(data))
        median_dwell[m,state_similarity[int(state)]] = np.median(data)
    
        
    
############# plot performance by target per state
fig = plt.figure(figsize=[16,12])
ax=[]
all_data = [[lh,rh],[le,re],[lm,rm]]
labels=['hit','error','no resp.']
for z in range(3):
    ax.append(plt.subplot(6,1,z+1))
    n=0
    for i in range(6):
        
        data=all_data[z][0][:,i]
        data=data[data==data]
        ax[z].bar(n,np.mean(data),yerr=sem(data),color=cols[i])
        ax[z].plot(n*np.ones(len(data))+np.random.randn(len(data))/10,data,'ko',markersize=4,clip_on=False)
        n=n+1
        
        data=all_data[z][1][:,i]
        data=data[data==data]
        ax[z].bar(n,np.mean(data),yerr=sem(data),color=cols[i])
        ax[z].plot(n*np.ones(len(data))+np.random.randn(len(data))/10,data,'ko',markersize=4,clip_on=False)
        n=n+2
    ax[z].set_ylabel('p(' + labels[z] + ')')
    ax[z].set_ylim([0,1])
    ax[z].set_xticks([0,1,3,4,6,7,9,10,12,13,15,16])
    ax[z].set_xticklabels('')
    ax[z].set_yticks(np.arange(0,1.1,.25),['0.0','','0.5','','1.0'])
ax[z].set_xticks([0,1,3,4,6,7,9,10,12,13,15,16])
ax[z].set_xticklabels(['L','R','L','R','L','R','L','R','L','R','L','R'])
ax[z].set_xlabel('Target port')

 

labels = ['Optimal\n(N mice = 13/13)','Disengaged\n(13/13)','Bias left\n(6/13)',
          'Avoid right\n(4/13)','Bias right\n(5/13)','Avoid left\n(5/13)']
xs = [.5,3.5,6.5,9.5,12.5,15.5]
for i in range(len(labels)):
    ax[0].text(xs[i],1.1,labels[i],ha='center',va='bottom')

ax[1].patch.set_alpha(0)
ax[2].patch.set_alpha(0)


for i in range(6):
    ax[0].add_patch(Rectangle([-.6+i*3,-2.4], 2.2, 3.4,color=cols[i],alpha=.2,clip_on=False,edgecolor = None,linewidth=0))

  
#### plot state occupancy
occ = np.array(all_state_occupancy)

ax.append(plt.subplot(position=[0.125,0.11,.25,.29]))
bplot = ax[-1].boxplot(occ,showfliers = False,patch_artist=True)
for i in range(4):
    data = occ[:,i]
    x=np.ones(len(data))*i+1+np.random.randn(len(data))/15
    ax[-1].plot(x,data,'ko',markersize=4,zorder=10)
ax[-1].set_ylabel('p(state)')
ax[-1].set_xticks(range(1,5))    
ax[-1].set_xticklabels(['Opt','Dis','Sub','Indeterminate'])    
ax[-1].set_ylim([0,.75])
ax[-1].set_yticks(np.arange(0,.8,.25))


a = '#679292'
b ='#c41111'
c ='#ffc589'
colors = [a,b,c,[.7,.7,.7]]
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

ax[-1].plot([1,3],[.8,.8],'k',clip_on=False,linewidth=2)
ax[-1].plot([1,1.9],[.75,.75],'k',clip_on=False,linewidth=2)
ax[-1].plot([2.1,3],[.75,.75],'k',clip_on=False,linewidth=2)

ax[-1].text(2,.83,'* * *',ha='center',va='top')
ax[-1].text(1.5,.78,'* * *',ha='center',va='top')
ax[-1].text(2.5,.78,'* *',ha='center',va='top')


#### plot cumulative distribution functions for dwell times

ax.append(plt.subplot(position=[0.45,0.11,.25,.29]))

l=[]
alpha = .2
basex= np.arange(0,500)
colors = [a,b,c,[.5,.5,.5]]
for state in range(4):
    mode_data = dwell_CDFs[:,state,:]

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
    p = Polygon(list(zip(x,y)),facecolor=colors[state],alpha=alpha)
    l.append(ax[-1].plot(basex[keep_ind],mean,'k',linewidth=2,color=colors[state])[0])
    ax[-1].add_patch(p)
    
ax[-1].set_ylabel('Cumulative probability')
ax[-1].set_xlabel('Dwell time (# trials)')
ax[-1].set_yticks(np.arange(0,1.1,.25))

## plot medians dwell times in inset
ax.append(fig.add_axes([0.56,0.16,.125,.15]))

mouse_med=np.full([len(hmm_trials_paths),4],np.nan)
for m in range(len(hmm_trials_paths)):
    for i in range(3):
        mouse_med[m,i] = np.where(dwell_CDFs[m,i,:]>.499)[0][0]
    mouse_med[m,3] = np.where(dwell_CDFs[m,3,:]>.499)[0][0]
        
        
bplot = ax[-1].boxplot([mouse_med[:,0],mouse_med[:,1],mouse_med[:,2],mouse_med[:,3]],showfliers=False,positions=[0,1,2,3],patch_artist=True)
for i in range(4):
    data = mouse_med[:,i]
    x=i*np.ones(len(data))+np.random.randn(len(data))*.05
    ax[-1].plot(x,data,'ko',markersize=3,zorder=10)
ax[-1].set_yscale('log')
ax[-1].set_ylabel('Median dwell\n(# trials)')
ax[-1].set_xticks([0,1,2,3])    
ax[-1].set_xticklabels(['Opt','Dis','Sub','Inde'])    
ax[-1].set_yticks([1,10,100])


a = '#679292'
b ='#c41111'
c ='#ffc589'
colors = [a,b,c,[.7,.7,.7]]
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


ax[-1].plot([0,1],[123,123],'k',clip_on=False)
ax[-1].text(.5,165,'* * *',ha='center',va='top')
ax[-1].plot([0,2],[190,190],'k',clip_on=False)
ax[-1].text(1,260,'* * *',ha='center',va='top')
ax[-1].set_ylim([1,120])

#### plot predictive accuracies
ax.append(plt.subplot(position=[0.775,0.11,.125,.29]))

predictive_accuracy = np.load(base_folder + 'misc\\choice_prediction_accuracy.npy')
p_acc_one = predictive_accuracy[:,1,0,:].mean(axis=1)*100
p_acc_best= predictive_accuracy[:,1,1,:].mean(axis=1)*100

bplot = ax[-1].boxplot([p_acc_one,p_acc_best],showfliers=False,patch_artist=True)

data = [p_acc_one,p_acc_best]
data = np.vstack(data)
for this_mouse in data.T:
    x=np.arange(2)+np.random.randn(2)*.05+1
    ax[-1].plot(x,this_mouse,color=[.7,.7,.7],zorder=10)
    ax[-1].plot(x,this_mouse,'ko',markersize=3,zorder=11)

   
a = [.5,.5,.5]
colors = [a,a]
b=bplot
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

ax[-1].set_ylabel('Test set choice prediction accuracy (%)')
ax[-1].set_xticks([1,2])
ax[-1].set_xticklabels(['GLM','GLM-HMM'])
ax[-1].plot([1,2],[94,94],'k',clip_on=False,linewidth=2)
ax[-1].text(1.5,95,'* * *',ha='center',va='center')
ax[-1].set_ylim([45,93])


for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)


ax[0].text(-2.5,1.25,'A',fontsize=30)
ax[0].text(-2.75,-3.25,'B',fontsize=30)
ax[0].text(5,-3.25,'C',fontsize=30)
ax[0].text(13,-3.25,'D',fontsize=30)

plt.savefig(base_folder + 'figures\\figure_2.pdf')
plt.savefig(base_folder + 'figures\\figure_2.png', dpi=300)