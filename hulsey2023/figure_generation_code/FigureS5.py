# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 12:07:37 2022

@author: admin
"""

import os
import numpy as np
import glob
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
import multiprocessing as mp
from uobrainflex.behavioranalysis.decoding import decode_states
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy import stats
from sklearn import preprocessing
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import colors
from matplotlib import cm
import ray


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt


def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])

cdict = {'red':   [(0.0,  inter_from_256(103), inter_from_256(103)),
                   (0.5,  1, 1),
                   (1.0,  inter_from_256(196), inter_from_256(196))],

         'green': [(0.0,  inter_from_256(146), inter_from_256(146)),
                   (0.5, 1, 1),
                   (1.0,  inter_from_256(17), inter_from_256(17))],

         'blue':  [(0.0,  inter_from_256(146), inter_from_256(146)),
                   (0.5,  1, 1),
                   (1.0,  inter_from_256(17), inter_from_256(17))]}


opt_dis_cm = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)

#############################
  
cdict = {'red':   [(0.0,  inter_from_256(103), inter_from_256(103)),
                   (0.5,  1, 1),
                   (1.0,  inter_from_256(255), inter_from_256(255))],

         'green': [(0.0,  inter_from_256(146), inter_from_256(146)),
                   (0.5, 1, 1),
                   (1.0,  inter_from_256(165), inter_from_256(165))],

         'blue':  [(0.0,  inter_from_256(146), inter_from_256(146)),
                   (0.5,  1, 1),
                   (1.0,  inter_from_256(30), inter_from_256(30))]}

opt_sub_cm = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)
###########################3
a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f,'k']


base_folder = 'D:\\Hulsey\\Hulsey_et_al_2023\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

@ray.remote
def fit_predict_map_SVM(train_feature,train_class,test_feature,test_class):
    SVM = svm.SVC(kernel='rbf')
    SVM.fit(train_feature,train_class)  
    accuracy = SVM.score(test_feature,test_class)
    
    response =SVM.decision_function(X_grid)
    response=response.reshape(xx0.shape)
    return accuracy,response

def scale(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


features = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10','movement','movement_std']
# for m in range(mouse_weights.shape[0]):

xx0, xx1 = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100))

X_grid = np.c_[xx0.ravel(), xx1.ravel()]

fig=plt.figure()
nKfold=5

process=[]
ms=[]
zs=[]
states=[]
folds=[]
run=[]
n_run=1
for m in range(len(hmm_trials_paths)):
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle = True)
    subject = os.path.basename(hmm_trials_paths[m])[:5]
    print(subject)
    
    
    all_trials = pd.DataFrame()
    for trials in hmm_trials:
        if np.unique(trials['hmm_state'])[0]==0:
            all_trials = all_trials.append(trials)
            
    all_trials = all_trials.query('post_hoc_pupil_std10==post_hoc_pupil_std10')
    all_trials = all_trials.query('face_std_10==face_std_10')
    all_trials = all_trials.query('running_std10==running_std10')
    all_trials = all_trials.query('post_hoc_pupil_diameter==post_hoc_pupil_diameter')
    all_trials = all_trials.query('face_energy==face_energy')
    all_trials = all_trials.query('running_speed==running_speed')
    all_trials['movement'] = stats.zscore(all_trials['face_energy']) + stats.zscore(all_trials['running_speed'])
    all_trials['movement_std'] = stats.zscore(all_trials['face_std_10']) + stats.zscore(all_trials['running_std10'])
    
    for feature in features:
        all_trials[feature] = all_trials[feature]-min(all_trials[feature])
        all_trials[feature] = all_trials[feature]/max(all_trials[feature])
    
    ax=[]
    for z in range(4):
        if z==0:
            best_features=[[0,1]]
        elif z==1:
            best_features=[[2,3]]
        elif z==2:
            best_features=[[4,5]]
        elif z==3:
            best_features=[[6,7]]
        
            
        
        
        opt_trials = all_trials.query('hmm_state==0')
        opt_trials = opt_trials.iloc[np.random.permutation(len(opt_trials))]
        dis_trials = all_trials.query('hmm_state==1')
        dis_trials = dis_trials.iloc[np.random.permutation(len(dis_trials))]
        sub1_trials = all_trials.query('hmm_state==2')
        sub1_trials = sub1_trials.iloc[np.random.permutation(len(sub1_trials))]
        sub2_trials = all_trials.query('hmm_state==3')
        sub2_trials = sub2_trials.iloc[np.random.permutation(len(sub2_trials))]
                
        
        all_data = [dis_trials]
        if len(sub1_trials)>10:
            all_data.append(sub1_trials)
        if len(sub2_trials)>10:
            all_data.append(sub2_trials)
        for state,data in enumerate(all_data):
            print(state)
            these_opt_trials=opt_trials.iloc[:len(data)]
            data=data.iloc[:len(these_opt_trials)]
            
           
            xopt = these_opt_trials[features[best_features[0][0]]].to_numpy()
            yopt = these_opt_trials[features[best_features[0][1]]].to_numpy()
            xdis = data[features[best_features[0][0]]].to_numpy()
            ydis = data[features[best_features[0][1]]].to_numpy()
        
            ind = round(len(data)/5*4)
            
            xopt = xopt.reshape(-1, 1)
            yopt = yopt.reshape(-1, 1)
            
            xdis = xdis.reshape(-1, 1)
            ydis = ydis.reshape(-1, 1)
            
            
            kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)  
            for ii, (train_index, test_index) in enumerate(kf.split(range(len(data)))):
                print(f"    kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
           
                for iii in range(n_run):
                    opt_train_feature = np.stack([xopt[train_index],yopt[train_index]])[:,:,0]
                    dis_train_feature = np.stack([xdis[train_index],ydis[train_index]])[:,:,0]
                    
                    train_feature = np.concatenate([opt_train_feature.T,dis_train_feature.T])
                    train_class = np.concatenate([np.zeros(len(train_index)),np.ones(len(train_index))])
                    
                    opt_test_feature = np.stack([xopt[test_index],yopt[test_index]])[:,:,0]
                    dis_test_feature = np.stack([xdis[test_index],ydis[test_index]]) [:,:,0]   
                    test_feature = np.concatenate([opt_test_feature.T,dis_test_feature.T])
                    test_class = np.concatenate([np.zeros(len(test_index)),np.ones(len(test_index))])
                    
                    if iii>0:
                        train_class=train_class[np.random.permutation(len(train_class))]
                    
                    process.append(fit_predict_map_SVM.remote(train_feature,train_class,test_feature,test_class))
                    ms.append(m)
                    zs.append(z)
                    states.append(state)
                    folds.append(ii)
                    run.append(iii)
                
                
                
all_response=np.full([len(hmm_trials_paths),3,4,nKfold,100,100],np.nan)
all_response2=np.full([len(hmm_trials_paths),3,4,nKfold,100,100],np.nan)
all_accuracy = np.full([len(hmm_trials_paths),3,4,nKfold],np.nan)

for i,p in enumerate(process):
    print(str(i) + ' of ' + str(len(process)))
    a,b = ray.get(p)
    all_accuracy[ms[i],states[i],zs[i],folds[i]]=a
    
    norm_b=np.array(b)
    norm_b[norm_b>1]=1
    norm_b[norm_b<-1]=-1
    all_response[ms[i],states[i],zs[i],folds[i],:,:]=b
    all_response2[ms[i],states[i],zs[i],folds[i],:,:]=norm_b
    

# np.save(base_folder+'individual_features_rbf_acc.npy',all_accuracy)
# np.save(base_folder+'individual_features_rbf_map.npy',all_response)
# np.save(base_folder+'individual_features_rbf_norm_map.npy',all_response2)


### import all feature combind rbf svm classifier results
files = glob.glob(base_folder + 'decoder_data\\*_rbf*SVM*acc*1000.npy')

all_feat_acc = []
for file in files:
    data = np.load(file)
    all_feat_acc.append(data)
all_feat_acc = np.stack(all_feat_acc)
all_feat_acc = all_feat_acc.mean(axis=-1)


z_scores=np.full([13,6,15],np.nan)
accuracies=np.full([13,6,15],np.nan)
for m in range(all_feat_acc.shape[0]):
    for s in range(all_feat_acc.shape[1]):
        for cond in range(all_feat_acc.shape[3]):
            shuffles = all_feat_acc[m,s,:,-1]
            if all(shuffles==shuffles):
                z=[]
                for run in range(all_feat_acc.shape[2]):
                    acc = all_feat_acc[m,s,run,cond]
                    z.append((acc-shuffles.mean())/shuffles.std())
                
                
                z_scores[m,s,cond]=np.mean(z)
                accuracies[m,s,cond]=all_feat_acc[m,s,:,cond].mean()



fig = plt.figure(figsize=[16,13])

row = [.1,.325]
column = [.15, .35, .55, .75]
size = .175
ax=[]
ax2=[]
features =['Pupil diameter','Face m.e.','Locomotion Speed','Movemenet index']
sbp=[]
for i in range(4):
    # ax.append(plt.subplot(4,4,i+9))
    if i<3:
        ax.append(plt.subplot(position = [column[i],row[1],size,size]))
    elif i==3:
        ax.append(plt.subplot(position = [column[i],row[1],size,size]))
    plt.imshow(all_response2[:,0,i,:,:,:].mean(axis=0).mean(axis=0),cmap=opt_dis_cm)
    plt.gca().invert_yaxis()    
    plt.clim([-1,1])
    plt.title(features[i])
    plt.xticks(np.arange(-.5,101,25),['','','','',''])
    if i==0:
        plt.yticks(np.arange(-.5,101,25),np.arange(0,1.1,.25))
        plt.ylabel('10 trial std. (norm)')
    else:
        plt.yticks(np.arange(-.5,101,25),['','','','',''])
    if i ==3:
        # ax2 = plt.subplot(position = [column[i]+.175,row[1],.15,.15])
        # plt.imshow(all_response2[:,0,i,:,:60,:10].mean(axis=0).mean(axis=0),cmap=opt_dis_cm)
        cbar = plt.colorbar(ax=ax)
        cbar.set_ticks(np.arange(-1,1.1,.5))
        cbar.ax.set_yticks([-1,-0.5,0,0.5,1])
        cbar.ax.tick_params() 
        # cbar.set_label('Average decision function')
        cbar.ax.get_yticks()

    # ax2.append(plt.subplot(4,4,i+13))
    ax2.append(plt.subplot(position = [column[i],row[0],size,size]))
    subs = np.vstack(all_response2[:,1:,i,:,:,:])
    subs=subs[np.unique(np.where(subs==subs)[0]),:,:,:]
    plt.imshow(subs.mean(axis=0).mean(axis=0),cmap=opt_sub_cm)
    plt.gca().invert_yaxis()    
    plt.clim([-1,1])
    plt.xticks(np.arange(-.5,101,25),np.arange(0,1.1,.25))
    plt.xlabel('Value (norm)')
    if i==0:
        plt.yticks(np.arange(-.5,101,25),np.arange(0,1.1,.25))
        plt.ylabel('10 trial std. (norm)')
    else:
        plt.yticks(np.arange(-.5,101,25),['','','','',''])
    if i ==3:
        cbar = plt.colorbar(ax=ax2)
        cbar.set_ticks(np.arange(-1,1.1,.5),)
        cbar.ax.set_yticks([-1,-0.5,0,0.5,1])
        cbar.ax.tick_params() 
        # cbar.set_label('Average decision function')
        cbar.ax.get_yticks()

# plt.subplot(4,4,9)
ax[0].text(95,95,'N mice = 13',ha='right',va='top')
ax[0].text(95,85,'n states = 13',ha='right',va='top')

# plt.subplot(4,4,13)
ax2[0].text(95,95,'N mice = 13',ha='right',va='top')
ax2[0].text(95,85,'n states = 19',ha='right',va='top')

# plt.subplot(4,4,12)
ax[3].text(133,113,'Disengaged',ha='center',va='top')
ax[3].text(133,-15,'Optimal',ha='center',va='bottom')
ax[3].text(120,50,'Average decision function',rotation=90,va='center',ha='center')

# plt.subplot(4,4,16)
ax2[3].text(133,113,'Sub-optimal',ha='center',va='top')
ax2[3].text(133,-15,'Optimal',ha='center',va='bottom')
ax2[3].text(120,50,'Average decision function',rotation=90,va='center',ha='center')

# plt.tight_layout()

fig

##
ax.append(plt.subplot(position=[.075,.625,.375,.325]))
cols=['k','m','#2278B5','#F57F20','#2FA148','r','k','#F57F20','#2FA148']
l=[]
idx=[0,2,3,4]
for i in idx:#range(z_scores.shape[-1]):
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]
    l.append(plt.scatter(these_accs,these_z,color=cols[i],alpha=.5,linewidth=0,edgecolor=None))
    
for i in idx:#range(z_scores.shape[-1]):
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]    
    plt.errorbar(these_accs.mean(),these_z.mean(),yerr=stats.sem(these_z),xerr=stats.sem(these_accs),color=cols[i],linewidth=4)
lg=[]

lg = ['All measures','Pupil diaemter','Face motion energy','Locomotion speed']
    
plt.legend(lg)
plt.ylabel('z-score')
plt.xlabel('Classification accuracy (%)')
plt.plot([.4,1],[1.98,1.98],'k--')
plt.plot([.5,.5],[0,12],'k--')
plt.ylim([0,11])
plt.xlim([.4,1])
plt.xticks(np.arange(.5,1.1,.25),np.arange(50,101,25))
plt.yticks(np.arange(0,11,2))
plt.text(.3,11,'A',fontsize=30)
plt.text(1.05,11,'B',fontsize=30)
plt.text(.3,-3,'C',fontsize=30)


##
ax.append(plt.subplot(position=[.535,.625,.375,.325]))
cols=['k','m','#2278B5','#F57F20','#2FA148','#2278B5','#F57F20','#2FA148','r','k','#F57F20','#2FA148']
l=[]
idx=[0,5,6,7]
for i in idx:#range(z_scores.shape[-1]):
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]
    l.append(plt.scatter(these_accs,these_z,color=cols[i],alpha=.5,linewidth=0,edgecolor=None))
    
for i in idx:#range(z_scores.shape[-1]):
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]    
    plt.errorbar(these_accs.mean(),these_z.mean(),yerr=stats.sem(these_z),xerr=stats.sem(these_accs),color=cols[i],linewidth=4)
lg=[]

lg = ['All measures','Shuffle pupil diaemter','Shuffle face motion energy','Shuffle locomotion speed']
    
plt.legend(lg)
plt.ylabel('z-score')
plt.xlabel('Classification accuracy (%)')
plt.plot([.4,1],[1.98,1.98],'k--')
plt.plot([.5,.5],[0,12],'k--')
plt.ylim([0,11])
plt.xlim([.4,1])
plt.xticks(np.arange(.5,1.1,.25),np.arange(50,101,25))
plt.yticks(np.arange(0,11,2))

 