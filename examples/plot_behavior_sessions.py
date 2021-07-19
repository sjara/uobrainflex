# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:12:19 2021

@author: admin
"""


from uobrainflex.nwb import loadbehavior as load
import matplotlib.pyplot as plt
import numpy as np
import pynwb


mice = ['BW016','BW017','BW018','BW019','BW020','BW021','BW026','BW027','BW028','BW029']
subject='BW016'
n=0
for mouse in mice:
    n=n+1
    subject = mouse
    sessions=[]
    for fileID in range(20210201,20210617):
        this_session = load.get_file_path(subject,str(fileID))
        if this_session:
            sessions.append(this_session)
            
            
    HR=list([])
    RR = list([])
    Hit_Balance=list([])
    stage = list([])
    for sess in sessions:
        trial_data, trial_labels = load.load_trial_data(sess)
        hit = np.where(trial_data['outcome'].values == trial_labels['outcome']['hit'])[0]
        
        np.where(trial_data['target_port'] == trial_labels['target_port']['right'])
        
        R_hit = len(np.where(np.in1d(hit, np.where(trial_data['target_port'] == trial_labels['target_port']['right'])))[0])
        L_hit = len(np.where(np.in1d(hit, np.where(trial_data['target_port'] == trial_labels['target_port']['left'])))[0])
        All_hit = len(hit)
        
        
        miss = len(np.where(trial_data['outcome'].values == trial_labels['outcome']['miss'])[0])
        IR = len(np.where(trial_data['outcome'].values == trial_labels['outcome']['incorrect_reject'])[0])
        ioObj = pynwb.NWBHDF5IO(sess, 'r', load_namespaces=True)
        this_data = ioObj.read()
        stage.append(int(this_data.lab_meta_data['metadata'].training_stage[1]))
        
        if All_hit != 0:
            Hit_Balance.append(R_hit/All_hit)
        else:
            Hit_Balance.append(.5)   
        if All_hit+miss !=0:
            HR.append(All_hit/(All_hit+miss))
        else:
            HR.append(0)
        RR.append(1-IR/len(trial_data))
        
        
    
    plt.figure(num=n)
    plt.plot(HR)
    plt.plot(Hit_Balance)
    plt.plot(np.divide(stage,20))
    plt.plot([0, len(HR)],[.66, .66],'k:')
    plt.plot([0, len(HR)],[.33, .33],'k:')
    plt.plot([0, len(HR)],[.75, .75],'k--')
    plt.title(subject)
    plt.xlabel('session')
    plt.legend(['Hit Rate','Hit Balance','Stage'])
    plt.draw()
    
   
##