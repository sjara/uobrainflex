"""
Create NWB file with behavior data from txt files.
"""

import sys
import os
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
import pynwb
import pandas as pd
import importlib

VERBOSE = True
EXTENSION_FILE = 'uobrainflex.namespace.yaml'

# FIXME: make the name of the behavior version and subject files as package setting.
BEHAVIOR_VERSION_FILE = 'Behavior_Version.txt' # 'Str_behavior_version.txt'
SUBJECT_FILE = 'Animal_ID.txt'


def read_schema(inputDir):
    """
    Load the schema used to convert txt files to NWB.
    """
    schemaStr = read_txt_data(inputDir, BEHAVIOR_VERSION_FILE, 'one_string')
    schemaName = 'schema_'+schemaStr
    schemaPath =  os.path.splitext(__name__)[0] + '.' + schemaName
    schema_behavior = importlib.import_module(schemaPath)
    return schema_behavior.schema


def get_nwb_basename(inputDir):
    """
    Return name of the output NWB file given an input directory with .txt files
    """
    schema = read_schema(inputDir)
    subject = read_txt_data(inputDir, schema['subject']['filename'], 'one_string')
    sessionStartTime = read_txt_data(inputDir, schema['session_start_time']['filename'], 'datetime')
    startTimeStr = sessionStartTime.strftime('%Y%m%dT%H%M%S')
    nwbBasename = make_nwb_basename(subject,startTimeStr)
    return nwbBasename


def make_nwb_basename(subject, startTimeStr):
    """
    Create name of the output NWB file.
    """
    nwbBasename = '{}_behavior_{}.nwb'.format(subject,startTimeStr)
    return nwbBasename


def get_subject(inputDir):
    """
    Return name of subject given an input directory with .txt files
    """
    return read_txt_data(inputDir, SUBJECT_FILE, 'one_string')


def read_txt_data(filedir, filename, datatype, fileprefix=''):
    """
    Reads data from a .txt file located in filedir.
    """
    fullpath = os.path.join(filedir, fileprefix+filename)
    try:
        if datatype == 'one_string':
            with open(fullpath, 'r') as file:
                data = file.readline().strip()
        elif datatype == 'float':
            data = np.loadtxt(fullpath, ndmin=1, dtype=float)
        elif datatype == 'int':
            data = np.loadtxt(fullpath, ndmin=1).astype(int)
        elif datatype == 'datetime':
            with open(fullpath, 'r') as file:
                dstr = file.readline().strip()
                dateNoTZ = datetime.strptime(dstr,'%Y-%m-%dT%H:%M:%S.%f')
                data = dateNoTZ.replace(tzinfo=tzlocal())
        elif datatype == 'timestamp':
            with open(fullpath, 'r') as file:
                datalines = file.read().splitlines()
            data = [datetime.strptime(dstr,'%y-%m-%dT%H:%M:%S.%f').timestamp() for dstr in datalines]
    except ValueError:
        print(f'ERROR: Format of data from file "{filename}" is not correct.')
        raise
    return data


def to_nwb(inputDir, outputDir, schema=None, outputFilename=None, verbose=True):
    """
    Load all behavior data from txt files and save a single NWB file.

    Args:
        inputDir (str): folder containing all txt files.
        outputDir (str): destination folder for the NWB file.
        schema (dict): dictionary containing definition of NWB file structure.
        outputFilename (str): if None it will be created automatically.
        verbose (bool): if True, show message for each step.

    Returns:
        nwbFile: NWB file object.
        outputFullfile: full path to the output NWB file.
    """

    if schema is None:
        schema = read_schema(inputDir)
    
    creationDate = datetime.now(tz=tzlocal())

    # -- Read general session parameters --
    if verbose:
        print('Loading session parameters...')
    subject = read_txt_data(inputDir, schema['subject']['filename'], 'one_string')
    experimenter = read_txt_data(inputDir, schema['experimenter']['filename'], 'one_string')
    sessionDescription = schema['session_description']
    institution = schema['institution']
    species = schema['species']
    sessionStartTime = read_txt_data(inputDir, schema['session_start_time']['filename'], 'datetime')

    # Date convention follows ISO 8601  (https://en.wikipedia.org/wiki/ISO_8601)
    startTimeStr = sessionStartTime.strftime('%Y%m%dT%H%M%S')
    uniqueID = '{}_{}'.format(subject,startTimeStr)

    if outputFilename is None:
        outputFilename = make_nwb_basename(subject, startTimeStr)

    subjectObj = pynwb.file.Subject(subject_id=subject, species=species)

    # -- Load extension (to enable metadata) --
    extensionDir = os.path.dirname(__file__)
    pynwb.load_namespaces(os.path.join(extensionDir,EXTENSION_FILE))
    LabMetaData_ext = pynwb.get_class('LabMetaData_ext', 'uobrainflex_metadata')
     
    # -- Initialize NWB file --
    nwbFile = pynwb.NWBFile(session_description = sessionDescription,  # required
                            identifier = uniqueID,                     # required
                            session_start_time = sessionStartTime,     # required
                            file_create_date = creationDate,
                            subject = subjectObj,
                            experimenter = experimenter,
                            institution = institution)

    # -- Lab metadata --
    metadata = {}
    for entryName, entryInfo in schema['metadata'].items():
        metadata[entryName] = read_txt_data(inputDir, entryInfo['filename'], 'one_string')    
    missing_fields = set(LabMetaData_ext.__nwbfields__).symmetric_difference(set(metadata))
    for entryName in missing_fields:
        metadata[entryName] = ''
    sessionMetadata = LabMetaData_ext(name='metadata', **metadata)
    nwbFile.add_lab_meta_data(sessionMetadata)

    # -- Store time series (running speed, pupil, etc) --
    for timeSeriesName, timeSeriesInfo in schema['acquisition'].items():
        if verbose:
            print('Loading {}...'.format(timeSeriesName))
        if 'timestamps' in timeSeriesInfo:
            timeSeriesTimestamps = read_txt_data(inputDir, timeSeriesInfo['timestamps'], 'float')
        elif 'rate' in timeSeriesInfo:
            raise ValueError('The use of sampling rate instead of timestamps '+\
                             'has not been implemented yet.')
        if 'filename' in timeSeriesInfo:
            timeSeriesData = read_txt_data(inputDir, timeSeriesInfo['filename'], 'float')
        else:
            timeSeriesData = np.zeros(len(timeSeriesTimestamps),dtype=int)
        timeSeriesObj = pynwb.TimeSeries(name=timeSeriesName, data=timeSeriesData,
                                         timestamps=timeSeriesTimestamps,
                                         unit=timeSeriesInfo['unit'],
                                         description=timeSeriesInfo['description'])
        nwbFile.add_acquisition(timeSeriesObj)

    # -- Create a dataframe with data from each trial (and specify columns for NWB file) --
    if verbose:
        print('Loading trial data...')
    trialDataFrame = pd.DataFrame()
    for fieldName,fieldInfo in schema['trials'].items():
        fieldDescription = fieldInfo['description']
        # -- If there is a map to interpret values, add it to the description
        if 'map' in fieldInfo:
            fieldDescription += " MAP:{}".format(str(fieldInfo['map']))
        if (fieldName != 'start_time') and (fieldName != 'stop_time'):
            nwbFile.add_trial_column(fieldName, description=fieldDescription)
        fieldData = read_txt_data(inputDir, fieldInfo['filename'], fieldInfo['dtype'])
        trialDataFrame[fieldName] = fieldData

    # -- Add trial data to NWB file --
    for index, row in trialDataFrame.iterrows():
        nwbFile.add_trial(**dict(row))

    # -- Save NWB file to disk --
    if verbose:
        print('Saving NWB file...')
    outputFullfile = os.path.join(outputDir,outputFilename)
    with pynwb.NWBHDF5IO(outputFullfile, 'w') as io:
        io.write(nwbFile)
    print('Saved {}'.format(outputFullfile))

    return nwbFile, outputFullfile
