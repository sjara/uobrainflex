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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Taken from online cookbook 21/10/28. https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    filter to be used for smoothing raw pupil, running speed, and whisker motion energy data
    
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def read_behavior_measures(nwbFileObj, speriod=0.001, measures=[], filt=True):
    """
    return continuous behavioral measures with specified sampling period from open nwb file.
    Args:
        nwbFileObj (object): an open nwbfile object.
        speriod (float): sample period in seconds.
        measures (list of strings): list of behavior measures to extract. 
    Returns:
        behavior_measures (DataFrame): Dataframe where index = samples, and columns = [measures, timestamps]
    """
    filt_params = {'whisker_energy': int(.5/speriod)+1, 'pupil_area': int(1/speriod)+1,
                   'running_speed': int(.2/speriod)+1,'pupil_diameter':int(1/speriod)+1}
    zero = {'whisker_energy': True, 'pupil_area': False, 'running_speed': False,'pupil_diameter':False}
    norm = {'whisker_energy': True, 'pupil_area': True, 'running_speed': False,'pupil_diameter':True}

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

    # filter and normalize signals if selected
    if filt:
        for key in measures:
            this_data = behavior_measures[key].values
            filt_data = savitzky_golay(this_data, filt_params[key], 2, deriv=0, rate=1)
            if norm[key]:
                if zero[key]:
                    filt_data = filt_data - np.nanmin(filt_data)
                filt_data = filt_data/np.nanmax(filt_data)
            behavior_measures[key] = filt_data
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


def fetch_trial_beh_measures(trial_data,behavior_measures):
    """
    Fetches behavior measures at moment of stimulus presentation
    
    Args:
        trial_data (dataframe): trial data loaded by load_trial_data function
        behavior_measures (dataframe): behavior measures loaded by load_behavior_measures function
            assumes 0.001 sample period used for behavior measures
    Returns:
        trial_data (DataFrame): trial data apended with behavior measures for each trial
    """
    measures = behavior_measures.columns
    measure_index = measures.values
    if any(measure_index == 'pupil_size'):
        measure_index[np.where(measure_index == 'pupil_diameter')[0][0]] = 'pupil_diameter'
            
    #loop through each trial
    for idx in trial_data.index:
        #get start sample
        trial_start = trial_data.loc[idx,'stimulus_time']
        start_sample = round(trial_start/.001) ####danger! this assumes ms sampling of behavior measures
        for measures_idx, key in enumerate(measures):
            #write behavior measures from start sample to trial_data
            trial_data.loc[idx,measure_index[measures_idx]] = behavior_measures.loc[start_sample,key]
        
    return trial_data

def fetch_trial_beh_events(trial_data,behavior_events):
    """
    Fetches behavior measures at instant of stimulus presentation
    
    Args:
        trial_data (dataframe): trial data loaded by load_trial_data function
        behavior_events (dict): behavior measures loaded by load_behavior_measures function
            assumes 0.001 sample period used for behavior measures
    Returns:
        trial_data (DataFrame): trial data apended with reaction times for each trial
    """
    # keys = behavior_events.keys()
    licks = np.sort(np.concatenate([behavior_events['licks_left'],behavior_events['licks_right']]))
    #loop through each trial
    rt=[]
    for idx in trial_data.index:
        #get start sample
        trial_start = trial_data.loc[idx,'stimulus_time']
        trial_licks = licks - trial_start*1000
        reaction_idx = np.where(trial_licks>0)[0]
        if len(reaction_idx)>0:
            rt.append(trial_licks[reaction_idx[0]])
        else:
            rt.append(np.nan)
    rt = np.array(rt)
    rt[np.where(rt>2250)[0]] = np.nan
    trial_data['reaction_time'] = rt
    return trial_data