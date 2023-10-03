import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold
from uobrainflex.behavioranalysis import flex_hmm
import matplotlib.pyplot as plt
import ray
from sklearn import svm

base_folder = input("Enter the main directory path") + '\\'
hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')
hmm_paths = glob.glob(base_folder + 'hmms\\*hmm.npy')
SVM = svm.SVC(probability = True)
all_measures = ['post_hoc_pupil_diameter','post_hoc_pupil_std10','face_energy','face_std_10','running_speed','running_std10']


#k fold taking 1/5 of data from a random chunk for each session
def stratified_kFold_random_order(class_index,nKfold):
    n_class = len(np.unique(class_index))
    fold_index = np.full([n_class,nKfold],int)
    for i in range(n_class):
        fold_index[i,:]=np.random.permutation(range(nKfold))
                                                    
    skf = StratifiedKFold(n_splits=nKfold)
    train_idx=[]
    test_idx=[]
    for iK, (train_index, test_index) in enumerate(skf.split(range(len(class_index)),class_index)):
        train_idx.append(train_index)
        test_idx.append(test_index)
        
    new_train_idx=[]
    new_test_idx=[]

    for i in range(nKfold):
        temp_train_index=[]
        temp_test_index =[]
        for j in range(n_class):
            
            this_class_idx = np.where(class_index==j)[0]
            
            temp_train = train_idx[fold_index[j,i]]
            temp_test = test_idx[fold_index[j,i]]
            temp_train_index.append(temp_train[np.isin(temp_train,this_class_idx)])
            temp_test_index.append(temp_test[np.isin(temp_test,this_class_idx)])
        new_train_idx.append(np.concatenate(temp_train_index))
        new_test_idx.append(np.concatenate(temp_test_index))
    return new_train_idx, new_test_idx
        

@ray.remote
def parallel_hmm_fit(n_states,train_inpts,train_choices):
    hmm = flex_hmm.choice_hmm_fit('test', n_states, [train_inpts], [train_choices])
    return hmm


# function to predict choice based on GLM weights 
def predict_choice(trial_inpt,state_params):
    state_params = params[predicted_trial_state]
    state_params = np.reshape(np.append(state_params,[0,0]),[3,2])

    choice_prob=np.full(3,np.nan)
    for choice in range(3):
        numerator = np.exp(sum((state_params[choice,:]-state_params[:,:])@trial_inpts))
        denominator = np.sum([np.exp(np.sum((state_params[choice,:]-state_params[:,:])@trial_inpts)) for choice in range(state_params.shape[0])])
        choice_prob[choice]=numerator/denominator
    return choice_prob.argmax()


ray.init()
val_only_ind = np.array([0,2,4])

n_mice = len(hmm_trials_paths)
nKfold=5
skf = StratifiedKFold(n_splits=nKfold)
max_states=7


predictive_accuracies=np.full([n_mice,2,2,nKfold],np.nan) #[mice,[test_ll, prediction accuracy],[1 state, best states],nKfold]

for m in range(n_mice):
    print('mouse ' + str(m+1) + ' of ' + str(n_mice))
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle = True)
    inpts = flex_hmm.get_inpts_from_hmm_trials(hmm_trials)
    true_choices = flex_hmm.get_true_choices_from_hmm_trials(hmm_trials)
    
    session_index=[]
    for i,inpt in enumerate(inpts):
        session_index.append(np.ones(len(inpt))*i)
        
    best_fit_hmm = np.load(hmm_paths[m],allow_pickle=True).tolist()
    best_states = best_fit_hmm.observations.params.shape[0]
    
    all_inpts = np.concatenate(inpts)
    all_choices = np.concatenate(true_choices)
            
    ###
    train_idx, test_idx = stratified_kFold_random_order(np.concatenate(session_index),nKfold)
    for iK in range(nKfold):
        print('mouse ' + str(m+1) + ' of ' + str(n_mice))
        print('fold ' + str(iK+1) + ' of ' + str(nKfold))
        train_index = train_idx[iK]
        test_index = test_idx[iK]
                
        train_inpts= all_inpts[train_index]
        train_choices = all_choices[train_index]
      
        test_inpts = all_inpts[test_index]
        test_choices = all_choices[test_index]        
        
        for z, n_states in enumerate([1,best_states]):
            print('mouse ' + str(m+1) + ' of ' + str(n_mice))
            print('fold ' + str(iK+1) + ' of ' + str(nKfold))
            print('state ' + str(n_states) + ' of ' + str(max_states))
            processes=[]
            for zz in range(5):
                processes.append(parallel_hmm_fit.remote(n_states,train_inpts,train_choices))
            results = [ray.get(p) for p in processes]
            hmms = np.array(results)   
    
            lls=[]
            for hmm in hmms:
                lls.append(hmm.log_probability(test_choices,test_inpts)/np.concatenate(test_inpts).shape[0]*np.log(2))
            hmm=hmms[np.argmax(lls)]
            
            #record test set log likelihood
            test_ll = hmm.log_probability(test_choices,test_inpts)/np.concatenate(test_inpts).shape[0]
            predictive_accuracies[m,0,z,iK]=test_ll
            

            #calculate and record test set choice prediction accuracy
            params = hmm.observations.params
            TM = hmm.transitions.params[0]
            
            predicted_choices=np.full(len(test_choices),np.nan)
            for i in range(len(test_inpts)):
                trial_inpts = test_inpts[i]
                if i==0:
                    predicted_trial_state = hmm.params[0][0].argmax()
                else:
                    posteriors = hmm.expected_states(data=test_choices[:i], input=test_inpts[:i])
                    state_posterior = posteriors[0][-1]
                    predicted_trial_state = TM[state_posterior.argmax()].argmax()
                            
                predicted_choices[i] = predict_choice(trial_inpts,params[predicted_trial_state])

            predictive_accuracies[m,1,z,iK]=len(np.where(np.concatenate(test_choices)==predicted_choices)[0])/len(predicted_choices)

        
np.save(base_folder + r'misc\choice_prediction_accuracy.npy', predictive_accuracies)





