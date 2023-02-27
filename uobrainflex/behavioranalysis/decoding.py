# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:49:53 2022

@author: admin
"""

import numpy as np
from sklearn import svm


def decode_states(condition,train_data,test_data,kernel='rbf'):
    train_states = train_data[1]
    train_measures = train_data[0]
    test_states = test_data[1]
    test_measures = test_data[0]
 
    permutation = np.random.permutation(len(train_states))
 
    
    if condition>18:
        train_states = train_states[permutation]
        
    #shuffle pupil values
    if condition==1:
        train_measures[:,0]=train_measures[permutation,0]
    #shuffle pupil variance
    if condition==2:
        train_measures[:,1]=train_measures[permutation,1]
    # shuffle both pupil value and variance
    if condition==3:
        train_measures[:,0]=train_measures[permutation,0]
        train_measures[:,1]=train_measures[permutation,1]                        
    
    # shuffle whisker value
    if condition==4:
        train_measures[:,2]=train_measures[permutation,2]
    #shuffle whisker variance
    if condition==5:
        train_measures[:,3]=train_measures[permutation,3]
    #shuffle both whisker value and variance
    if condition==6:
        train_measures[:,2]=train_measures[permutation,2]
        train_measures[:,3]=train_measures[permutation,3]        
        
        
    # shuffle run value
    if condition==7:
        train_measures[:,4]=train_measures[permutation,4]
    #shuffle run variance
    if condition==8:
        train_measures[:,5]=train_measures[permutation,5]
    #shuffle both run value and variance
    if condition==9:
        train_measures[:,4]=train_measures[permutation,4]
        train_measures[:,5]=train_measures[permutation,5]    
        
    # shuffle all but pupil value
    if condition==10:
        for shufind in [1,2,3,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but pupil variance
    if condition==11:
        for shufind in [0,2,3,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but pupil val and var
    if condition==12:
        for shufind in [2,3,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
        
        
    # shuffle all but whisk value
    if condition==13:
        for shufind in [0,1,3,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but whisk variance
    if condition==14:
        for shufind in [0,1,2,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but whisk val and var
    if condition==15:
        for shufind in [0,1,4,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]                        
    
        
    # shuffle all but run value
    if condition==16:
        for shufind in [0,1,2,3,5]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but run variance
    if condition==17:
        for shufind in [0,1,2,3,4]:
            train_measures[:,shufind]=train_measures[permutation,shufind]
    #shuffle all but run val and var
    if condition==18:
        for shufind in [0,1,2,3]:
            train_measures[:,shufind]=train_measures[permutation,shufind]  
    
    
    SVM = svm.SVC(kernel=kernel)
    SVM.fit(train_measures,train_states.ravel())  
    accuracy = SVM.score(test_measures,test_states)
    
    return [condition, SVM, accuracy]