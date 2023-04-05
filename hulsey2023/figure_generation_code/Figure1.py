# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:38:05 2023

@author: admin
"""


import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import sem
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from uobrainflex.nwb import loadbehavior as load
import ssm
from matplotlib.patches import Rectangle
from uobrainflex.behavioranalysis import flex_hmm
from scipy import stats

a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f]

base_folder = 'E:\\22-10-19-Data\\hmm_trials_nans2\\'
hmm_trials_paths = glob.glob(base_folder + '*hmm_trials.npy')
nwb_paths = glob.glob(base_folder + '*nwbfilepaths.npy')
hmm_paths = glob.glob(base_folder + '*hmm.npy')


fig = plt.figure(figsize=[16,12])
ax = fig.add_axes([0.1, .1, 0.8, .8])
# Create figure and subplots
# row1_height = .225
row1_height = .3
# row1_y = .725
row1_y = .675
#panel A
axA=plt.subplot(position=[.06,row1_y,.3,row1_height])

#panel B
axB=plt.subplot(position=[.41,row1_y+.01+row1_height/2,.3,row1_height/2])

#planelC
axC=plt.subplot(position=[.41,row1_y-.02,.3,row1_height/2])


#planelD
axD=plt.subplot(position=[.8,row1_y+.05,.1,row1_height-.075])


#planelE
axE=plt.subplot(position=[.175,.325,.8,.25])


row3_height = .225
row3_y = .05
#planelF
axF=plt.subplot(position=[.025,.00,.225,.285])

#planelG
axG=plt.subplot(position=[.3,row3_y,.11,row3_height])

#planelH
axH=plt.subplot(position=[.5,row3_y,.1,row3_height-.01])

#planelI
axI=plt.subplot(position=[.625,row3_y,.1,row3_height-.01])

#planelJ
axJ=plt.subplot(position=[.75,row3_y,.1,row3_height-.01])

#planelK
axK=plt.subplot(position=[.875,row3_y,.1,row3_height-.01])


#planel A
axA=plt.subplot(position=[.06,row1_y,.3,row1_height])
rig_file = 'E:\\22-10-19-Data\\hmm_trials_nans2\\code\\behavior_rig_schematic.bmp'
rig_image_data = plt.imread(rig_file)

axA.imshow(rig_image_data)
axA.text(175,260,'Speaker',ha='right',va='center')

axA.text(150,600,'Mouse',ha='right',va='center')
axA.arrow(175,600,375-190,500-635, width=2, color="k",
          head_width=35, head_length=3.5, overhang=10)

axA.text(450,1000,'Wheel',ha='right',va='center')
axA.arrow(400,975,450-400,775-975, width=2, color="k",
          head_width=35, head_length=3.5, overhang=10)

axA.text(1300,600,'Rotary\nencoder',ha='center',va='top')
axA.arrow(1175,625,900-1175,625-625, width=2, color="k",
          head_width=35, head_length=3.5, overhang=10)

axA.text(1300,375,'Reward\nports',ha='center',va='top')
axA.arrow(1170,400,615-1150,350-400, width=2, color="k",
          head_width=35, head_length=3.5, overhang=10)

axA.text(1100,15,'Camera',ha='center',va='top')
axA.text(725,25,'White & IR\nLEDs',ha='center',va='top')
axA.arrow(650,125,625-650,180-125, width=2, color="k",
          head_width=35, head_length=3.5, overhang=10)


axA.text(350,100,'Screen',ha='center',va='center')

# axA.text(0,0,'A',fontsize=30,ha='right',va='top')

axA.get_xlim()


axA.axis('off')


#panel B
stimulus_example_file = 'E:\\22-10-19-Data\\hmm_trials_nans2\\code\\stimulus_examples.bmp'
stim_example = plt.imread(stimulus_example_file)

axB.imshow(stim_example)
axB.plot([25,345],[65,65],'k')
axB.plot([25,345],[130,130],'k')
axB.plot([25,25],[130,127],'k')
axB.plot([105,105],[130,127],'k')
axB.plot([185,185],[130,127],'k')
axB.plot([265,265],[130,127],'k')
axB.plot([345,345],[130,127],'k')
axB.text(185,140,'Stimulus value',ha='center',va='top')
axB.text(25,140,'-1',ha='center',va='top')
axB.text(345,140,'1',ha='center',va='top')
axB.text(100,0,'Left-Target',ha='center',va='center')
axB.text(270,0,'Right-Target',ha='center',va='center')


axB.axis('off')
# axB.text(0,0,'B',fontsize=30,ha='right',va='top')

#panel C
trial_structure_file = 'E:\\22-10-19-Data\\hmm_trials_nans2\\code\\trial_structure.bmp'
trial_structure = plt.imread(trial_structure_file)

axC.imshow(trial_structure)
axC.text(65,60,'5+/-2 s\nITI',ha='center',va='center')
axC.text(200,45,'1.2 s\n Stimulus',ha='center',va='center')
axC.text(270,45,'1 s',ha='center',va='center')
axC.text(230,82,'Response',ha='center',va='center')
axC.text(50,110,'Lick',ha='left',va='center')

axC.axis('off')

# axB.text(0,165,'C',fontsize=30,ha='right',va='top')


######gather example subject data
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
    
    missing_values = np.unique(np.where(psycho1!=psycho1)[2])
    missing_values2 = np.unique(np.where(psycho2!=psycho2)[2])
    missing_vals = np.unique(np.concatenate([missing_values,missing_values2]))
    
    psycho1 = np.delete(psycho1,missing_vals,axis=2)
    psycho2 = np.delete(psycho2,missing_vals,axis=2)
    
    for state in psycho2:
        similarity=[]
        for state2 in psycho1:
            similarity.append(np.sum(abs(state-state2)))
        all_similarity.append(similarity)
    return np.array(all_similarity)




m=2
mouse_path = hmm_trials_paths[m]
subject = os.path.basename(mouse_path)[:5]
hmm_trials = np.load(mouse_path,allow_pickle=True)
posterior_probs = flex_hmm.get_posterior_probs_from_hmm_trials(hmm_trials)
dwell_data = flex_hmm.get_dwell_times(hmm_trials)

sess = 12
sess = 10
trial_data = hmm_trials[sess]
trial_probs = posterior_probs[sess]
session_dwell = dwell_data[sess]

m=2
############ plot psychometrics for states

line_style=[':',(0, (3, 3, 1, 3)),'-']
line_style=[':',(0, (3, 3, 1, 3)),'-']
line_style=['-','-','-']
marker=['<','>','o']
state_names = ['Optimal','Disengaged','Left Bias','Avoid Right','Right Bias','Avoid Left']
all_state_params  = np.full([len(hmm_paths),6,2,2],np.nan) 
all_state_psychos = np.full([len(hmm_paths),6,3,6],np.nan)


all_trials = pd.DataFrame()
no_hmm_trials=[]
 
for trials in hmm_trials:
    no_hmm = trials.copy()
    no_hmm['hmm_state']=0
    no_hmm_trials.append(no_hmm)
    # all_trials = all_trials.append(trials)
    
    

psychos, xvals = get_psychos(hmm_trials)
no_hmm_psychos, no_hmm_xvals = get_psychos(no_hmm_trials)

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


del_ind = np.unique(np.where([psychos!=psychos])[-1])
psycho = np.delete(psychos,del_ind,axis=2)

if psycho.shape[2]>6:
    psycho=np.delete(psycho,[1,-2],axis=2)

similarity = compare_psychos(sort_psychos,psycho)
state_similarity = np.nanargmin(similarity,axis=0)
state_similarity = np.nanargmin(similarity,axis=1)
print(state_similarity)

while len(np.unique(state_similarity)) != len(state_similarity):
    states, counts = np.unique(state_similarity,return_counts=True)
    for confused_state in states[np.where(counts>1)[0]]:
        confused_ind = np.where(state_similarity == confused_state)[0]
        confused_state_similarity = similarity[confused_ind,:]
        replace_state_ind = np.argmax(confused_state_similarity[:,confused_state])
        replace_state_similarity = similarity[confused_ind[replace_state_ind],:]
        repalcement_state = np.argsort(replace_state_similarity)[1]
        if repalcement_state == confused_state:
            repalcement_state = np.argsort(replace_state_similarity)[2]
        state_similarity[confused_ind[replace_state_ind]]=repalcement_state
    print('fixed with ' + str(state_similarity))



#Panel D


all_psychos = np.full([len(no_hmm_trials),3,6],np.nan)

state=0
for i, session in enumerate(no_hmm_trials):
    optimal_trials = session.query('hmm_state==@state')
    if len(optimal_trials)>20:
        no_hmm_psychos,no_hmm_xvals = flex_hmm.get_trials_psychos(session.query('hmm_state==@state'))
        if len(xvals)==6:
            all_psychos[i,:,:] = no_hmm_psychos
        
all_psychos = np.delete(all_psychos,np.unique(np.where(all_psychos!=all_psychos)[0]),axis=0)


for i in range(all_psychos.shape[1]):
    data = np.nanmean(all_psychos[:,i,:],axis=0)
    data = np.concatenate([np.array([data[0]]),data,np.array([data[-1]])])
    x = np.concatenate([np.array([-1.1]),xvals,np.array([1.1])])
    
    axD.plot(x,data,color='k',linestyle=line_style[i],linewidth=3,zorder=-1)

for i in range(all_psychos.shape[1]):
    data = np.nanmean(all_psychos[:,i,:],axis=0)
    error = sem(all_psychos[:,i,:],axis=0)
                 
    for j in range(len(data)):
        axD.errorbar(xvals[j],data[j],error[j],color='k',linewidth=3)
        
axD.plot(-1.1,np.nanmean(all_psychos[:,0,0]),color='k',marker=marker[0],markersize=15,markeredgecolor='k',markeredgewidth=3)
axD.plot(1.1,np.nanmean(all_psychos[:,0,-1]),color='k',marker=marker[0],markersize=15,markeredgecolor='k',markeredgewidth=3)
axD.plot(-1.1,np.nanmean(all_psychos[:,1,0]),color='k',marker=marker[1],markersize=15,markeredgecolor='k',markeredgewidth=3)
axD.plot(1.1,np.nanmean(all_psychos[:,1,-1]),color='k',marker=marker[1],markersize=15,markeredgecolor='k',markeredgewidth=3)

axD.plot(-1.1,np.nanmean(all_psychos[:,2,0]),color='k',marker=marker[2],markersize=15,markeredgecolor='k',markeredgewidth=3)
axD.plot(1.1,np.nanmean(all_psychos[:,2,-1]),color='k',marker=marker[2],markersize=15,markeredgecolor='k',markeredgewidth=3)

# all_state_psychos[m,state_ID,:,:] = np.nanmean(all_psychos,axis=0)

axD.set_xticks([-1,-.5,0,.5,1],[-1,'',0,'',1])
axD.set_xlabel('Stimulus value')
axD.set_ylim([-.1,1.1])
axD.set_xlim(-1.4,1.4)
# plt.title(state_names[state] +'\n n sessions = ' + str(all_psychos.shape[0]))
axD.set_title('All trials' + '\nn sessions = ' + str(all_psychos.shape[0]))
        


axD.set_ylabel('p(choice)')
axD.set_yticks(np.arange(0,1.1,.25))
axD.set_yticklabels(['0','','0.5','','1'])
for spine in ['top','right']:
   axD.spines[spine].set_visible(False)

axD.text(-22.5,1.3,'A',fontsize=30,ha='right',va='top')
axD.text(-12.3,1.3,'B',fontsize=30,ha='right',va='top')
axD.text(-12.3,.3,'C',fontsize=30,ha='right',va='top')
axD.text(-3,1.3,'D',fontsize=30,ha='right',va='top')

#Panel E
###### 80-100% of posteriors patch for state
###### x axis time

axE=plt.subplot(position=[.2,.325,.775,.25])

hand=[]
lg=[]

t_zero = trial_data['start_time'].iloc[0]
t = trial_data['start_time'].values-t_zero
for i, probs in enumerate(trial_probs):
    axE.plot(t,probs,color=cols[state_similarity[i]],linewidth=3,clip_on=False)
 
if len(session_dwell.index)>0:   
    for idx in session_dwell.index: 
        start_t = t[session_dwell.loc[idx,'first_trial']]
        dur = t[session_dwell.loc[idx,'last_trial']]-start_t
        axE.add_patch(Rectangle((start_t,.8), dur, .2,color=cols[state_similarity[int(session_dwell.loc[idx,'state'])]],alpha=.2))
    for idx in range(0,int(max(session_dwell['state']))+1):
        hand.append(ax.add_patch(Rectangle((0,0),0,0,color=cols[state_similarity[int(session_dwell.loc[idx,'state'])]],alpha=.2)))
        lg.append('state ' + str(idx+1))
        
axE.add_patch(Rectangle((t[0],1.15), t[-1], .095,color=[.7,.7,.7],alpha=.2,clip_on=False))
axE.add_patch(Rectangle((t[0],1.05), t[-1], .095,color=[.7,.7,.7],alpha=.4,clip_on=False))

for i in range(len(trial_data)):
    choice = trial_data.iloc[i]['choice']
    outcome = trial_data.iloc[i]['outcome']
    inpt = trial_data.iloc[i]['inpt']
    if choice == 2:
        choice=1.2
        if inpt<0:
            choice=1.225
        else:
            choice=1.175
        
        m='s'
    if choice ==0:
        choice = 1.3
        m='o'
    if choice == 1:
        choice = 1.1
        m='o'
    if outcome == 1:
        m=m+'k'
    elif outcome==2:
        m=m+'r'
    else:
        m=m+'b'
        
    if outcome==16:
        outcome=3
    if inpt<0:
        m='ko'
    else:
        m='ro'
        
    if abs(inpt) == .2:
        msize = 4
    elif abs(inpt) == .6:
        msize = 5
    else:
        msize= 6 
        
    axE.plot(t[i],1.4-outcome/10+np.random.randn()/85,m,alpha = .6,markersize=msize,clip_on=False)




axE.set_xlim([0,4342])


# axE.set_title(trial_data['file_name'].iloc[0])
axE.set_ylabel('p(state)')
axE.set_yticks([0,.2,.4,.6,.8,1,1.1,1.2,1.3],[0,.2,.4,.6,.8,1,'Miss','Error','Hit'])
axE.set_ylim([0,1])
# axE.set_xticks(fontsize=15)
for spine in ['top','right','bottom']:
    axE.spines[spine].set_visible(False)
axE.set_xticks([])
axE.set_ylabel('p(state)')
axE.text(-20,1.1,'No resp.',ha='right',va='center')
axE.text(-20,1.2,'Error',ha='right',va='center')
axE.text(-20,1.3,'Hit',ha='right',va='center')



for i,state in enumerate(np.unique(session_dwell['state']).astype(int)):
    axE.add_patch(Rectangle((-800,.2*(3-i)), 300, .15,color=cols[state_similarity[state]],alpha=.2,clip_on=False))
axE.text(-900,.2,'HMM state', rotation=90)

axE.plot(-900,1.2,'ko',alpha = .6,markersize = 7,clip_on=False)
axE.plot(-800,1.2,'ko',alpha = .6,markersize = 5,clip_on=False)
axE.plot(-700,1.2,'ko',alpha = .6,markersize = 3,clip_on=False)
axE.plot(-600,1.2,'ro',alpha = .6,markersize = 3,clip_on=False)
axE.plot(-500,1.2,'ro',alpha = .6,markersize = 5,clip_on=False)
axE.plot(-400,1.2,'ro',alpha = .6,markersize = 7,clip_on=False)
axE.text(-650,1.275,'Stimulus value',horizontalalignment='center')
axE.text(-900,1.1,'L',horizontalalignment='center',verticalalignment='center')
axE.text(-400,1.1,'R',horizontalalignment='center',verticalalignment='center')

axE.plot([0,t[-1]],[.8,.8],'k--',linewidth=.5)
axE.plot([-800,-500],[.95,.95],'k',linewidth=5,clip_on=False)
axE.text(-650,.85,'5 minutes',horizontalalignment='center',verticalalignment='center')

axE.text(100,.5,'Left bias',rotation=90,ha='center',va='center')
axE.text(280,.5,'Disengaged',rotation=90,ha='center',va='center')
axE.text(1100,.5,'Optimal',rotation=90,ha='center',va='center')
axE.text(1942,.5,'Right bias',rotation=90,ha='center',va='center')
axE.text(3230,.5,'Indeterminate',rotation=90,ha='center',va='center')
# axE.axis('on') 
# axE.spines['bottom'].set_visible(True)

axE.text(-1050,1.3,'E',horizontalalignment='center',verticalalignment='center',fontsize=30)


#panel F
GLM_GMM_cartoon_file = 'E:\\22-10-19-Data\\hmm_trials_nans2\\code\\GLM-HMM_cartoon.png'
GLM_GMM_cartoon = plt.imread(GLM_GMM_cartoon_file)

axF.imshow(GLM_GMM_cartoon)
axF.text(970,130,'State 1',ha='right',va='bottom')
axF.text(1075,250,'.98',ha='left',va='top')
axF.text(1075,630,'.01',ha='left',va='bottom')
axF.text(720,690,'.01',ha='left',va='top')

axF.text(270,1450,'State 2',ha='left',va='top')
axF.text(650,1100,'.02',ha='left',va='bottom')
axF.text(475,885,'.03',ha='right',va='bottom')
axF.text(130,1307,'.95',ha='right',va='bottom')

axF.text(1230,1170,'State\nn',ha='center',va='center')
axF.text(1360,1450,'.97',ha='right',va='top')
axF.text(975,1275,'.01',ha='right',va='top')
axF.text(1000,870,'.02',ha='right',va='top')

axF.axis('off') 

axE.text(-1050,-.15,'F',horizontalalignment='center',verticalalignment='center',fontsize=30)



#Panel G

def plateu(data,threshold=.001):
    ind = np.where(np.diff(data)>threshold)[0]
    ind[np.argmax(data[ind+1])]
    return ind[np.argmax(data[ind+1])]+1

base_folder = 'E:\\22-10-19-Data\\hmm_trials_nans2\\'
hmm_trials_paths = glob.glob(base_folder + '*hmm_trials.npy')
MAP_test_files = glob.glob(base_folder + '*MAP_test*')
subject = "BW046"
LL = np.load(glob.glob(base_folder+'\\'+subject+'*MAP_test*')[-1])
example_LL = LL.mean(axis=-1).max(axis=0)
num_states = plateu(example_LL)
example_predictive_accuracy = np.load(base_folder + 'BW046_predictive_accuracy.npy')


axG.plot(example_LL-example_LL[0],'r') 
# ax.set_xlabel("N states", fontsize = 12)
ticks=['1','','','4','','','7']
axG.set_xticks(range(0,len(example_LL)))
axG.set_xticklabels(ticks)
axG.plot(num_states,example_LL[num_states]-example_LL[0],'rs')
axG.set_xlabel('# HMM states')
axG.set_ylabel("Δ test LL\n(bits per trial)",
              color="red")
axG.set_yticks(np.arange(0,.31,.05))
axG.set_yticklabels(['0.0','','0.1','','0.2','','0.3'])
ax2=axG.twinx()
ax2.set_position(axG.get_position())
ax2.plot(example_predictive_accuracy*100,'k')
ax2.set_ylabel("Predictive accuracy",rotation=270,ha='center',va='bottom')
ax2.set_yticks(np.arange(70,91,5))
ax2.set_yticklabels(['70','','80','','90'])
ax2.spines['top'].set_visible(False)
axG.spines['top'].set_visible(False)
axG.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
axG.tick_params(axis='y', colors='red')

axE.text(250,-.15,'G',horizontalalignment='center',verticalalignment='center',fontsize=30)

#Panel H
axs = [axH,axI,axJ,axK]
n=1
for state, state_ID in enumerate(state_similarity):
    n=n+1
    #plot psycho

    all_psychos = np.full([len(hmm_trials),3,6],np.nan)
    for i, session in enumerate(hmm_trials):
        optimal_trials = session.query('hmm_state==@state')
        if len(optimal_trials)>20:
            psychos,xvals = flex_hmm.get_trials_psychos(session.query('hmm_state==@state'))
            if len(xvals)==6:
                all_psychos[i,:,:] = psychos
    
    xvals = np.unique(session['inpt'])
    all_psychos = np.delete(all_psychos,np.unique(np.where(all_psychos!=all_psychos)[0]),axis=0)
    

    for i in range(all_psychos.shape[1]):
        data = np.nanmean(all_psychos[:,i,:],axis=0)
        data = np.concatenate([np.array([data[0]]),data,np.array([data[-1]])])
        x = np.concatenate([np.array([-1.1]),xvals,np.array([1.1])])
        
        axs[state].plot(x,data,color=cols[state_ID],linestyle=line_style[i],linewidth=3,zorder=-1)
    
    for i in range(all_psychos.shape[1]):
        data = np.nanmean(all_psychos[:,i,:],axis=0)
        error = sem(all_psychos[:,i,:],axis=0)
                     
        for j in range(len(data)):
            axs[state].errorbar(xvals[j],data[j],error[j],color='k',linewidth=3)
            
    axs[state].plot(-1.1,np.nanmean(all_psychos[:,0,0]),color=cols[state_ID],marker=marker[0],markersize=15,markeredgecolor='k',markeredgewidth=3)
    axs[state].plot(1.1,np.nanmean(all_psychos[:,0,-1]),color=cols[state_ID],marker=marker[0],markersize=15,markeredgecolor='k',markeredgewidth=3)
    axs[state].plot(-1.1,np.nanmean(all_psychos[:,1,0]),color=cols[state_ID],marker=marker[1],markersize=15,markeredgecolor='k',markeredgewidth=3)
    axs[state].plot(1.1,np.nanmean(all_psychos[:,1,-1]),color=cols[state_ID],marker=marker[1],markersize=15,markeredgecolor='k',markeredgewidth=3)

    axs[state].plot(-1.1,np.nanmean(all_psychos[:,2,0]),color=cols[state_ID],marker=marker[2],markersize=15,markeredgecolor='k',markeredgewidth=3)
    axs[state].plot(1.1,np.nanmean(all_psychos[:,2,-1]),color=cols[state_ID],marker=marker[2],markersize=15,markeredgecolor='k',markeredgewidth=3)
    
    

    if state == 0 :
        axs[state].set_ylabel('p(choice)')
        axs[state].set_yticks(np.arange(0,1.1,.25))
        axs[state].set_yticklabels(['0','','0.5','','1'])
    else:
        axs[state].set_yticks(np.arange(0,1.1,.25))
        axs[state].set_yticklabels(['','','','',''])
    
    axs[state].set_xticks([-1,-.5,0,.5,1],[-1,'',0,'',1])
    axs[state].set_xlabel('Stimulus value')
    axs[state].set_ylim([-.1,1.1])
    axs[state].set_xlim(-1.4,1.4)
    # plt.title(state_names[state] +'\n n sessions = ' + str(all_psychos.shape[0]))
    if n==2:
        axs[state].set_title(state_names[state_ID] + '\nn Sessions = ' + str(all_psychos.shape[0]) + '/34')
    else:
        axs[state].set_title(state_names[state_ID] + '\nn = ' + str(all_psychos.shape[0]) + '/34')

    for spine in ['top','right']:
        axs[state].spines[spine].set_visible(False)

axE.text(1500,-.15,'H',horizontalalignment='center',verticalalignment='center',fontsize=30)


plt.rcParams.update({'font.size': 15,'font.weight': 'normal','axes.titlesize': 15})


plt.savefig('C:\\Users\\admin\\Desktop\\figure dump\\Figure_1.png', dpi=300)