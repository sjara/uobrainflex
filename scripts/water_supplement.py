# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:27:50 2021

@author: admin
"""

import os
from datetime import datetime
import numpy as np
import sys

if len(sys.argv)>1:
    subjects = sys.argv[1:]
else:
    subjects = ['no_subject']

folder = 'D:\\Brain_Behavior_Suite\\'

today = datetime.today().strftime('%y%m%d')
today2 = datetime.today().strftime('%Y-%m-%d')
pump_txt = 'Pump_Time.txt'
stage_txt = 'Training_Stage.txt'
outcome_txt = 'Trial_Outcome.txt'

for mouse in subjects:
    rewards = []
    sessions = set(os.listdir(folder+mouse))
    for session in sessions:
        if session.find(today) != -1:
            pump_file = os.path.join(folder,mouse,session,pump_txt)
            pump_times = np.loadtxt(pump_file, ndmin=1, dtype=float)  
        
            stage_file = os.path.join(folder,mouse,session,stage_txt)
            with open(stage_file, 'r') as file:
                stage = file.readline().strip()
                
            if stage != 'S1':
                rewards.append(len(pump_times))
            else:
                outcome_file = os.path.join(folder,mouse,session,outcome_txt)
                outcome = np.loadtxt(outcome_file, ndmin=1, dtype=float)
                rewards.append(len(np.where(outcome==1)[0]))
    water = sum(rewards)*3
    supplement = (1000-water)/1000
    if np.less(supplement,0):
        supplement=  0
    print(str(len(rewards)) + ' sessions found for ' + mouse + ' with ' + str(water) + ' ul of water collected on ' + today2 +'\n'
          + 'Supplement ' + str(supplement) + ' ml of water\n')               