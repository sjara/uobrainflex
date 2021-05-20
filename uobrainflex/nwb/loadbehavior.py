"""
Functions and classes for loading behavior data from NWB files.
"""

import os
import glob
import pynwb
import numpy as np
import pandas as pd
from uobrainflex import config

behavior_measures_keylist = ['pupil_size','whisker_energy','running_speed']
behavior_events_keylist = ['reward_right','reward_left','licks_left','licks_right']

def get_file_path(subject, fileID):
    """   
    Return full path to an NWB file based on matching fileID. 
    Returns first file if multiple are found.
    Returns '' if no file with matching ID is found.
    Args:
        subject (string): name of subject.
        fileID (string): search input for nwb file. ideally 
    Returns:
        nwb_file_path (string): full path to an NWB file
    """
    dataDir = config['BEHAVIOR']['cloud_storage_path']
    subject_dir = os.path.join(dataDir, subject)
    nwbfilepath = glob.glob(os.path.join(subject_dir,f'{subject}*{fileID}*.nwb'))
    if not nwbfilepath:
        nwbfilepath = ''
    else:
        nwbfilepath=nwbfilepath[0]
    return nwbfilepath

def load_trial_data(nwbfilepath):
    """
    Return trial information and labels map to read enumerate intergers.
    Args:
        nwbfilepath (string): full path to an NWB file.
    Returns:
        trial_data (DataFrame): index of trial number, columns of relevant data
        trial_labels (dict): dictionary of dictionaries containing mapping meanings to constants
    """
    # load nwb file
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    this_data = ioObj.read()
    # extract trial_data
    trial_data = this_data.trials.to_dataframe()
    # fetch and incorporate stage ****this will be very buggy
    # stage = np.full(len(trial_data),1)*int(this_data.lab_meta_data['metadata'].training_stage[1])
    # trial_data['stage'] = stage
      
    #create dict of trial_labels stored in trials column descriptions
    trial_labels = {}
    for key in this_data.trials.colnames:
        this_map = this_data.trials[key].description
        # if there is a '{' in this description, evaluate from that character to end of description to make dictionary for this key
        if this_map.find('{') != -1:
            this_map = this_map[(this_map.find('{')):]   
            trial_labels[key] = eval(this_map)     
    return trial_data, trial_labels


def load_behavior_measures(nwbfilepath, speriod=0.001, measures=[]):
    """
    return continuous behavioral measures from nwb file.
    Args:
        nwbfilepath (string): full path to an NWB file.
        speriod (float): sample period in seconds.
        measures (list of strings): list of behavior measures to extract. 
    Returns:
        behavior_measures (DataFrame): Dataframe where index = sample and columns = [measures, timestamps]
    """
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    this_data = ioObj.read()
    
    if not measures:
        measures = behavior_measures_keylist
    
    # calculate number of periods to index with based on sample rate and length of longest measures data set
    max_time=np.full(len(measures),np.NaN)
    for idx, key in enumerate(measures):
        key_max = this_data.get_acquisition(key).timestamps[:]
        if key_max.size > 0:
            max_time[idx] = (key_max[-1]/speriod).round(0).astype(int)           
    
    # crate index and corresponding timestamps from start of session
    df_index = int(max(max_time))+1
    timestamps = pd.date_range(this_data.session_start_time, periods=df_index-1, freq='{}ms'.format(speriod*1000))
    
    behavior_measures = pd.DataFrame(index = range(df_index))    
    # extract all acquisition time series
    for key in measures:
        ts_data = this_data.get_acquisition(key).data[:]
        ts_time = this_data.get_acquisition(key).timestamps[:]
        idx = (ts_time/speriod).round(0).astype(int) # divide by sample period and round to find appropriate index for this measure
        behavior_measures.loc[idx,key]=ts_data       # fit data to new sampling
    # interpolate dataframe to upsample and make full 
    behavior_measures = behavior_measures.iloc[:max(max_time).astype(int),:].interpolate(method = 'linear') 
    behavior_measures['timestamps'] = timestamps.values
    return behavior_measures

def load_behavior_events(nwbfilepath, speriod=0.001, events=[]):
    """
    return behavioral events from nwb file.
    Args:
        nwbfilepath (string): full path to an NWB file.
        events (list of strings): list of behavior events to extract.
    Returns:
        behavior_events (dict of nparrays): dictionary of arrays of event times rounded to ms. NOTE: MAY NOT ALIGN WITH BEHAVIOR_MEASURES SAMPLING
    """
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    this_data = ioObj.read()
    
    if not events:
        events = behavior_events_keylist
    
    max_time=np.full(len(events),np.NaN)
    for idx, key in enumerate(events):
        key_max = this_data.get_acquisition(key).timestamps[:]
        if key_max.size > 0:
            max_time[idx] = (key_max[-1]/speriod).round(0).astype(int)           
    
    # crate index and corresponding timestamps from start of session
    df_index = int(max(max_time))+1
    timestamps = pd.date_range(this_data.session_start_time, periods=df_index, freq='{}ms'.format(speriod*1000))
    
    behavior_events = {}
    for key in events:
        ts_time = this_data.get_acquisition(key).timestamps[:]
        idx = (ts_time/speriod).round(0).astype(int)
        behavior_events[key] = timestamps[idx]          # will output as absolute timestamps
        #behavior_events[key] = idx                     # will output as index corresponding to behavior_events
        
    return behavior_events


