import os
import numpy as np
import glob
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from scipy import stats
from matplotlib import colors


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size']= 15
rcParams['font.weight']= 'normal'
rcParams['axes.titlesize']= 15
rcParams['ytick.labelsize']= 15
rcParams['xtick.labelsize']= 15
import matplotlib.pyplot as plt


# set color maps for plotting
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
###########################
a = '#679292'
b ='#c41111'
c ='#ffc589'
d ='#ffa51e'
e ='#f26c2a'
f = '#ec410d'
cols = [a,b,c,d,e,f,'k']
##########################


base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')

features = ['post_hoc_pupil_diameter','post_hoc_pupil_std10']
nKfold=5

# load example mouse adta
m=6
hmm_trials = np.load(hmm_trials_paths[m],allow_pickle = True)
subject = os.path.basename(hmm_trials_paths[m])[:5]

all_trials = pd.DataFrame()
for trials in hmm_trials:
    if np.unique(trials['hmm_state'])[0]==0:
        all_trials = all_trials.append(trials)
   
## exclude any trials with nan data and normalize
all_trials = all_trials.query('post_hoc_pupil_std10==post_hoc_pupil_std10')

for feature in features:
    all_trials[feature] = all_trials[feature]-min(all_trials[feature])
    all_trials[feature] = all_trials[feature]/max(all_trials[feature])

#randomize order of trials and select equal proportions of each state
opt_trials = all_trials.query('hmm_state==0')
opt_trials = opt_trials.iloc[np.random.permutation(len(opt_trials))]
dis_trials = all_trials.query('hmm_state==1')
dis_trials = dis_trials.iloc[np.random.permutation(len(dis_trials))]

these_opt_trials=opt_trials.iloc[:len(dis_trials)]
                     
xopt = these_opt_trials[features[0]].to_numpy()
yopt = these_opt_trials[features[1]].to_numpy()
xdis = dis_trials[features[0]].to_numpy()
ydis = dis_trials[features[1]].to_numpy()

xopt = xopt.reshape(-1, 1)
yopt = yopt.reshape(-1, 1)

xdis = xdis.reshape(-1, 1)
ydis = ydis.reshape(-1, 1)

# segment data for cross validated classification
kf = KFold(n_splits=nKfold, shuffle=True, random_state=None)  
for ii, (train_index, test_index) in enumerate(kf.split(range(len(dis_trials)))):
    print(f"    kfold {ii} TRAIN:", len(train_index), "TEST:", len(test_index))
   
opt_train_feature = np.stack([xopt[train_index],yopt[train_index]])[:,:,0]
dis_train_feature = np.stack([xdis[train_index],ydis[train_index]])[:,:,0]

train_feature = np.concatenate([opt_train_feature.T,dis_train_feature.T])
train_class = np.concatenate([np.zeros(len(train_index)),np.ones(len(train_index))])

opt_test_feature = np.stack([xopt[test_index],yopt[test_index]])[:,:,0]
dis_test_feature = np.stack([xdis[test_index],ydis[test_index]]) [:,:,0]   
test_feature = np.concatenate([opt_test_feature.T,dis_test_feature.T])
test_class = np.concatenate([np.zeros(len(test_index)),np.ones(len(test_index))])


# fit SVM classifier 
SVM = svm.SVC(kernel='rbf')
SVM.fit(train_feature,train_class)  

### generate figure
fig = plt.figure(figsize=[16,5])
ax=[]
ax.append(plt.subplot(position=[.06,.125,.25,.75]))

### Panel A - map and plot SVM decision function
xx0, xx1 = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100))

X_grid = np.c_[xx0.ravel(), xx1.ravel()]
X=X_grid

response =SVM.decision_function(X_grid)
response=response.reshape(xx0.shape)
response[response>1]=1
response[response<-1]=-1
color_mesh = ax[-1].pcolormesh(response,cmap=opt_dis_cm)

ax[-1].set_aspect('equal')
ax[-1].set_xlim([0,100])
ax[-1].set_ylim([0,100])
ax[-1].set_xticks(np.arange(0,101,25))
ax[-1].set_xticklabels(np.arange(0,101,25))
ax[-1].set_yticks(np.arange(0,101,25))
ax[-1].set_yticklabels(np.arange(0,101,25))

p = ax[-1].scatter(opt_train_feature[0,:]*100,opt_train_feature[1,:]*100,color=cols[0],alpha =1,s=10)
p.set_edgecolor('k')
p.set_linewidth(1)
p = ax[-1].scatter(dis_train_feature[0,:]*100,dis_train_feature[1,:]*100,color=cols[1],alpha =1,s=10)
p.set_edgecolor('k')
p.set_linewidth(1)
p=ax[-1].scatter(opt_test_feature[0,:]*100,opt_test_feature[1,:]*100,color=cols[0])
p.set_edgecolor('k')
p.set_linewidth(1)
p=ax[-1].scatter(dis_test_feature[0,:]*100,dis_test_feature[1,:]*100,color=cols[1])
p.set_edgecolor('k')
ax[-1].set_ylabel('10 trial pupil std. (norm)')
ax[-1].set_xlabel('Pupil diameter (norm)')
ax[-1].text(117,105,'Disengaged',ha='center',va='bottom')
ax[-1].text(117,-5,'Optimal',ha='center',va='top')
ax[-1].text(109,50,'Decision function',ha='right',va='center',rotation=90)


###  make color bar
ax.append(plt.subplot(position=[.335,.125,.02,.75]))

cbar = fig.colorbar(color_mesh,ax[1])
cbar.set_ticks(np.arange(-1,1.1,.5),)
cbar.ax.set_yticks([-1,-0.5,0,0.5,1])       

### Panel B - show z scoring & accuracy
#import all feature combined rbf svm classifier results
files = glob.glob(base_folder + 'decoder_data\\*_rbf*SVM*acc*1000.npy')

all_feat_acc = []
for file in files:
    data = np.load(file)
    all_feat_acc.append(data)
all_feat_acc = np.stack(all_feat_acc)
all_feat_acc = all_feat_acc.mean(axis=-1) # average across folds

# z score against shuffle data
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
                accuracies[m,s,cond]=all_feat_acc[m,s,:,cond].mean() # average across 1000 runs

# ax.append(plt.subplot(1,3,2))
ax.append(plt.subplot(position=[.455,.125,.225,.75]))

m=6
shuffles=all_feat_acc[m,0,:,-1]
ax[-1].hist(shuffles,color=[.7,.7,.7])
acc=all_feat_acc[m,0,:,2].mean()
plt.plot([acc,acc],[0,250],'k',linewidth=3,color = '#2278B5')
ax[-1].set_xticks(np.arange(.25,1.1,.25))
ax[-1].set_xticklabels(np.arange(25,101,25))
this_z=(acc-shuffles.mean())/shuffles.std()
ax[-1].text(.825,225,'z = ' + str(this_z.round(2)),color='#2278B5',ha='right')
ax[-1].set_ylabel('Shuffle count')
ax[-1].set_xlabel('Classification accuracy (%)')


### Panel C - show all comparisons
ax.append(plt.subplot(position=[.75,.125,.225,.75]))

cols=['k','m','#2278B5','#F57F20','#2FA148','r','k','#F57F20','#2FA148']
l=[]
idx=[0,5,2] ## [all measures, shuffle all but movement index, shuffle all but pupil]
idx=[0,1,2,3,4,5] 
for i in idx: # scatter plot all data points
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]
    l.append(plt.scatter(these_accs,these_z,color=cols[i],alpha=.5,linewidth=0,edgecolor=None))
    
for i in idx: # show mean and sem as crosshairs
    these_accs = accuracies[:,:,i]
    these_z = z_scores[:,:,i]
    
    these_accs=these_accs[these_accs==these_accs]
    these_z=these_z[these_z==these_z]    
    plt.errorbar(these_accs.mean(),these_z.mean(),yerr=stats.sem(these_z),xerr=stats.sem(these_accs),color=cols[i],linewidth=4)
lg=[]


lg = ['All measures','Movement index','Pupil diameter']
    
plt.legend(lg)
plt.plot(accuracies[m,0,2],z_scores[m,0,2],'*',color=cols[2],markersize=12)
plt.ylabel('z-score')
plt.xlabel('Classification accuracy (%)')
plt.plot([.4,1],[1.98,1.98],'k--')
plt.plot([.5,.5],[0,12],'k--')
plt.ylim([0,11])
plt.xlim([.4,1])
plt.xticks(np.arange(.5,1.1,.25),np.arange(50,101,25))
plt.yticks(np.arange(0,11,2))
############

for spine in ['top','right']:
    for axs in ax:
        axs.spines[spine].set_visible(False)
        
ax[0].text(-25,102,'A',fontsize=30)
ax[0].text(140,102,'B',fontsize=30)
ax[0].text(267,102,'C',fontsize=30)


plt.savefig(base_folder + 'figures\\figure_5.pdf')
plt.savefig(base_folder + 'figures\\figure_5.png', dpi=300)