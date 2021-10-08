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
        fileID (string): search input for nwb file. ideally a unique identifier
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

def get_recent_sessions(subject,num_sessions):
    """
    Returns specified number of most recent session filepaths uploaded as nwb files

    Args:
        subject (string): name of subject
        num_sessions (int): number of session filepaths to return
    Returns:
        session_paths(list): list of strings of most recent behavioral sessions
    """
    
    dataDir = config['BEHAVIOR']['cloud_storage_path']
    subject_dir = dataDir + '\\' + subject
    nwbfiles = glob.glob(os.path.join(subject_dir,'*.nwb'))
    
    if len(nwbfiles)>num_sessions:
        nwbfiles = nwbfiles[len(nwbfiles)-num_sessions:]
    
    return nwbfiles

def load_nwb_file(nwbfilepath):
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    nwbFileObj = ioObj.read()
    return nwbFileObj

def read_trial_data(nwbFileObj):
    """
    Return trial information and labels map to read enumerated intergers from open nwb file.
    Args:
        nwbFileObj (object): an open nwbfile object.
    Returns:
        trial_data (DataFrame): index of trial number, columns of relevant data
        trial_labels (dict): dictionary of dictionaries containing mapping meanings to constants
    """
    # extract trial_data
    trial_data = nwbFileObj.trials.to_dataframe()
    # fetch and incorporate stage ****this will be very buggy
    # stage = np.full(len(trial_data),1)*int(nwbFileObj.lab_meta_data['metadata'].training_stage[1])
    # trial_data['stage'] = stage
      
    #create dict of trial_labels stored in trials column descriptions
    trial_labels = {}
    for key in nwbFileObj.trials.colnames:
        this_map = nwbFileObj.trials[key].description
        # if there is a '{' in this description, evaluate from that character to end of description to make dictionary for this key
        if this_map.find('{') != -1:
            this_map = this_map[(this_map.find('{')):]   
            trial_labels[key] = eval(this_map)    
    return trial_data, trial_labels


def read_behavior_measures(nwbFileObj, speriod=0.001, measures=[]):
    """
    return continuous behavioral measures with specified sampling period from open nwb file.
    Args:
        nwbFileObj (object): an open nwbfile object.
        speriod (float): sample period in seconds.
        measures (list of strings): list of behavior measures to extract. 
    Returns:
        behavior_measures (DataFrame): Dataframe where index = samples, and columns = [measures, timestamps]
    """

    if not measures:
        keys = nwbFileObj.acquisition.keys()
        measures = list(set(keys).symmetric_difference(set(behavior_events_keylist)))        
    
    # calculate number of periods to index with based on sample rate and length of longest measures data set
    max_time=np.full(len(measures),np.NaN)
    for idx, key in enumerate(measures):
        key_max = nwbFileObj.get_acquisition(key).timestamps[:]
        if key_max.size > 0:
            max_time[idx] = (key_max[-1]/speriod).round(0).astype(int)           
    
    # crate index and corresponding timestamps from start of session
    df_index = int(max(max_time))
    start_time = nwbFileObj.session_start_time.timestamp()
    end_time = round(start_time+df_index*speriod,3)
    timestamps=np.arange(int(start_time*1000), int(end_time*1000),int(1000*speriod))/1000  
    
    behavior_measures = pd.DataFrame(index = range(df_index+1))    
    # extract all acquisition time series
    for key in measures:
        ts_data = nwbFileObj.get_acquisition(key).data[:]
        ts_time = nwbFileObj.get_acquisition(key).timestamps[:]
        idx = (ts_time/speriod).round(0).astype(int) # divide by sample period and round to find appropriate index for this measure
        idx = idx[:ts_data.shape[0]]
        behavior_measures.loc[idx,key]=ts_data       # fit data to new sampling
    # interpolate dataframe to upsample and make full 
    behavior_measures = behavior_measures.iloc[:max(max_time).astype(int),:].interpolate(method = 'linear') 
    behavior_measures['timestamps'] = timestamps
    return behavior_measures

def read_behavior_events(nwbFileObj, speriod=0.001, events=[]):
    """
    return behavioral events at input sampling precision from open nwb file.
    Args:
        nwbFileObj (object): an open nwbfile object.
        speriod (float): sample period in seconds.
        events (list of strings): list of behavior events to extract.
    Returns:
        behavior_events (dict of nparrays): dictionary of arrays of event times in UTC timestamps
        behavior_events_index (dict of nparrays): dictionary of arrays of event times indexed from start time by speriod.
    """
    
    if not events:
        events = behavior_events_keylist
    
    max_time=np.full(len(events),np.NaN)
    for idx, key in enumerate(events):
        key_max = nwbFileObj.get_acquisition(key).timestamps[:]
        if key_max.size > 0:
            max_time[idx] = (key_max[-1]/speriod).round(0).astype(int)
        else:
            max_time[idx]=0
        
    df_index = int(max(max_time))+1
    start_time = nwbFileObj.session_start_time.timestamp()
    end_time = round(start_time+df_index*speriod,3)
    timestamps=np.arange(int(start_time*1000), int(end_time*1000),int(1000*speriod))/1000

    behavior_events = {}
    behavior_events_index={}
    for key in events:
        ts_time = nwbFileObj.get_acquisition(key).timestamps[:]
        idx = (ts_time/speriod).round(0).astype(int)
        behavior_events[key] = timestamps[idx]
        behavior_events_index[key] = idx
        
    return behavior_events, behavior_events_index


def load_trial_data(nwbfilepath):
    """
    Opens an nwb file, then returns trial information and labels map to read enumerate intergers.
    Args:
        nwbfilepath (string): full path to an NWB file.
    Returns:
        trial_data (DataFrame): index of trial number, columns of relevant data
        trial_labels (dict): dictionary of dictionaries containing mapping meanings to constants
    """
    # load nwb file
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    nwbFileObj = ioObj.read()
    
    trial_data, trial_labels = read_trial_data(nwbFileObj)
    
    ioObj.close()
    return trial_data, trial_labels


def load_behavior_measures(nwbfilepath, speriod=0.001, measures=[]):
    """
    Opens nwb file, then returns behavioral measures at indicated sapmle period.
    Args:
        nwbfilepath (string): full path to an NWB file.
        speriod (float): sample period in seconds.
        measures (list of strings): list of behavior measures to extract. 
    Returns:
        behavior_measures (DataFrame): Dataframe where index = samples, and columns = [measures, timestamps]
    """
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    nwbFileObj = ioObj.read()
    
    behavior_measures = read_behavior_measures(nwbFileObj, speriod, measures)

    ioObj.close()
    return behavior_measures

def load_behavior_events(nwbfilepath, speriod=0.001, events=[]):
    """
    Opens nwb file, then returns behavioral events with specified sampling precision.
    Args:
        nwbfilepath (string): full path to an NWB file.
        speriod (float): sample period in seconds.
        events (list of strings): list of behavior events to extract.
    Returns:
        behavior_events (dict of nparrays): dictionary of arrays of event times in UTC timestamps
        behavior_events_index (dict of nparrays): dictionary of arrays of event times indexed from start time by speriod.
    """
    ioObj = pynwb.NWBHDF5IO(nwbfilepath, 'r', load_namespaces=True)
    nwbFileObj = ioObj.read()
    
    behavior_events, behavior_events_index = read_behavior_events(nwbFileObj, speriod, events)
        
    ioObj.close()
    return behavior_events, behavior_events_index

