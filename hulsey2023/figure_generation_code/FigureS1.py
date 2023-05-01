# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:46:37 2022

@author: admin
"""

import os
import numpy as np
import glob
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt


def plateu(data,threshold=.001):
    ind = np.where(np.diff(data)>threshold)[0]
    ind[np.argmax(data[ind+1])]
    return ind[np.argmax(data[ind+1])]+1

base_folder = 'D:\\Hulsey\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')


n_states=[]
fig = plt.figure(figsize=[18,14])
best_states=[]
for m in range(len(hmm_trials_paths)):
    plt.subplot(3,5,m+1)
    subject = os.path.basename(hmm_trials_paths[m])[:5]   
    this_hmm = np.load(glob.glob(base_folder+'hmms\\'+subject+'*hmm.npy')[-1],allow_pickle= True)
    this_hmm = this_hmm.tolist()
    old_n_states = this_hmm.observations.params.shape[0]
    
    LL = np.load(glob.glob(base_folder+'n_states\\'+subject+'*MLE_test*')[-1])
    LL = LL.mean(axis=-1).max(axis=0)
    LL=LL-LL[0]
    MAP_num_states = plateu(LL)+1
    plt.plot(range(1,len(LL)+1),LL,'k')
    best_states.append(plt.plot(MAP_num_states,LL[MAP_num_states-1],'ks'))
       
    LL = np.load(glob.glob(base_folder+'n_states\\'+subject+'*MAP_test*')[-1])
    LL = LL.mean(axis=-1).max(axis=0)
    LL=LL-LL[0]
    MLE_num_states = np.argmax(LL)+1    
    plt.plot(range(1,len(LL)+1),LL,'r')
    best_states.append(plt.plot(MLE_num_states,LL[MLE_num_states-1],'rs')) 
      
    if np.isin(m,[0,5,10]):
        plt.yticks(np.arange(0,.35,.1))
        plt.ylabel('Δ test LL\n(bits per trial)')
    else:
        plt.yticks(np.arange(0,.35,.1),'')
       
    if np.isin(m,[10,11,12]):
        plt.xticks(range(1,len(LL)+1))
        plt.xlabel('# HMM states')
    else:
        plt.xticks(range(1,len(LL)+1),'')

    plt.title(subject)
    plt.ylim([0,.375])
    
    n_states.append(max([MAP_num_states,MLE_num_states]))

plt.subplot(3,5,m+3)
plt.hist(n_states,bins=[2.5,3.5,4.5,5.5],color='k',width=.8)
plt.ylabel('mice (count)')
plt.xlabel('# states')
plt.yticks(range(0,8,2),range(0,8,2))
plt.xticks(range(3,6),range(3,6))


plt.subplot(6,5,24)
plt.plot(range(1,len(LL)+1),LL,'k')
plt.plot(range(1,len(LL)+1),LL,'r')
plt.ylim([100,101])
plt.xticks([])
plt.yticks([])
plt.legend(['MLE','MAP'],title = 'Test set LL',loc = 'center')
fig.axes[-1].spines['bottom'].set_visible(False)
fig.axes[-1].spines['left'].set_visible(False)
plt.subplot(6,5,29)
plt.plot(MLE_num_states,LL[MLE_num_states-1],'ks')
plt.plot(MLE_num_states,LL[MLE_num_states-1],'rs')
plt.ylim([100,101])
plt.xticks([])
plt.yticks([])
plt.legend(['MLE','MAP'],title = 'Best # states',loc = 'center')
fig.axes[-1].spines['bottom'].set_visible(False)
fig.axes[-1].spines['left'].set_visible(False)

for spine in ['top','right']:
    for axs in fig.axes:
        axs.spines[spine].set_visible(False)

plt.savefig(base_folder + 'figures\\figure_S1.pdf')
plt.savefig(base_folder + 'figures\\figure_S1.png', dpi=300)