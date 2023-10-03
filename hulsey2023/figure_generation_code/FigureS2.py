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

base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')


n_states=[]
fig = plt.figure(figsize=[16,18])
best_states=[]
subjects=[]
for m in range(len(hmm_trials_paths)):
    plt.subplot(4,5,m+1)
    subject = os.path.basename(hmm_trials_paths[m])[:5]   
    subjects.append(subject)
    this_hmm = np.load(glob.glob(base_folder+'hmms\\'+subject+'*hmm.npy')[-1],allow_pickle= True)
    this_hmm = this_hmm.tolist()
    old_n_states = this_hmm.observations.params.shape[0]
    
    LL = np.load(glob.glob(base_folder+'n_states\\'+subject+'*MLE_test*')[-1])
    LL = LL.mean(axis=-1).max(axis=0)
    LL=LL-LL[0]
    MLE_num_states = plateu(LL)+1
    plt.plot(range(1,len(LL)+1),LL,'k')
    best_states.append(plt.plot(MLE_num_states,LL[MLE_num_states-1],'ks'))
       
    LL = np.load(glob.glob(base_folder+'n_states\\'+subject+'*MAP_test*')[-1])
    LL = LL.mean(axis=-1).max(axis=0)
    LL=LL-LL[0]
    MAP_num_states = np.argmax(LL)+1    
    plt.plot(range(1,len(LL)+1),LL,'r')
    best_states.append(plt.plot(MAP_num_states,LL[MAP_num_states-1],'rs')) 
      
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

plt.subplot(4,5,m+3)
plt.hist(n_states,bins=[2.5,3.5,4.5,5.5],color='k',width=.8)
plt.ylabel('mice (count)')
plt.xlabel('# states')
plt.yticks(range(0,8,2),range(0,8,2))
plt.xticks(range(3,6),range(3,6))


plt.subplot(8,5,24)
plt.plot(range(1,len(LL)+1),LL,'k')
plt.plot(range(1,len(LL)+1),LL,'r')
plt.ylim([100,101])
plt.xticks([])
plt.yticks([])
plt.legend(['MLE','MAP'],title = 'Test set LL',loc = 'center')
fig.axes[-1].spines['bottom'].set_visible(False)
fig.axes[-1].spines['left'].set_visible(False)
plt.subplot(8,5,29)
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

plt.tight_layout()
all_state_similarity = np.load(r'D:\Hulsey\Hulsey_et_al_2023\misc\state_similarities.npy',allow_pickle = True)
states_idx = np.full([13,6],'',dtype=str)

for i, states in enumerate(all_state_similarity):
    for this_state in states:
        states_idx[i,this_state]='x'

states_idx = np.vstack([['Optimal','Disengaged','Bias left','Avoid right','Bias right','Avoid left'],states_idx])

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
axs = plt.subplot(20, 1,16)
# ax = fig.add_axes([0.1, .1, 0.8, .8])
# axs = plt.subplots(position=[.2,.2,.5,.5])
columns = np.concatenate([['State'],subjects])
axs.axis('tight')
axs.axis('off')
table = axs.table(cellText=states_idx.T, colLabels=columns, cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1,3)
table.auto_set_column_width(col=0) # Provide integer list of columns to adjust
table.auto_set_column_width(col=range(14)) # Provide integer list of columns to adjust


# plt.savefig(base_folder + 'figures\\figure_S1_v2.pdf')
# plt.savefig(base_folder + 'figures\\figure_S1_v2.png', dpi=300)