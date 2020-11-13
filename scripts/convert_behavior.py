"""
Converts behavior data to NWB format, copies data to cloud storage,
and registers the data onto a DataJoint database.

Many of these steps require your configuration file (uobrainflex.conf)
to indicate the correct paths for cloud storage and DataJoint.

How to use this script:
- Run by specifying the folder containing .txt data files:
  python convert_behavior.py /data/labview/testmouse_201028_141726/

- Run it without arguments to open a GUI to select the data folder:
  python convert_behavior.py
"""

import os
import sys
import tkinter.filedialog
import tkinter
from uobrainflex import config
from uobrainflex.nwb import savebehavior
from uobrainflex.utils import cloudstorage
# NOTE: Modules related to DataJoint are imported below to avoid
#       trying to connect to the database unless it's necessary.

import importlib
importlib.reload(savebehavior)  # Useful during development

    
# -- Get input directory --
if len(sys.argv)>1:
    inputDir = sys.argv[1]
else:
    labview_data_path = config['BEHAVIOR']['labview_data_path']
    dialogTitle = 'Choose one txt file from the data folder to process'
    rootWin = tkinter.Tk()
    rootWin.withdraw()
    oneDataFilePath = tkinter.filedialog.askopenfilename(initialdir=labview_data_path,
                                                         title=dialogTitle)
    if not oneDataFilePath:
        raise ValueError('You need to select a valid file.')
    inputDir = os.path.split(oneDataFilePath)[0]

# -- Check if output file and folder exist --
subject = savebehavior.get_subject(inputDir)
subjectOutputDir = os.path.join(config['BEHAVIOR']['nwb_data_path'], subject)
nwbFilename = savebehavior.get_nwb_basename(inputDir)
nwbFullpath = os.path.join(subjectOutputDir,nwbFilename)
CONVERT_TO_NWB = True
if not os.path.isdir(subjectOutputDir):
    print('Created a new folder for this subject: {}'.format(subjectOutputDir))
    os.mkdir(subjectOutputDir)
elif os.path.isfile(nwbFullpath):
    overwriteStr = input('File {} exists. Overwrite? [Y/n] '.format(nwbFullpath))
    CONVERT_TO_NWB = (overwriteStr.lower()=='y') or not overwriteStr

# -- Ask user if save and register --
saveToCloudStr = input('Save data to cloud? [Y/n] ')
SAVE_TO_CLOUD = (saveToCloudStr.lower()=='y') or not saveToCloudStr

registerStr = input('Register session with DataJoint? [Y/n] ')
REGISTER_WITH_DATAJOINT = (registerStr.lower()=='y') or not registerStr

if CONVERT_TO_NWB:
    nwbFile, nwbPath = savebehavior.to_nwb(inputDir, subjectOutputDir)
    assert nwbPath==nwbFullpath  # Check that it saved in the correct location

if SAVE_TO_CLOUD:
    cloudPath = cloudstorage.save_behavior_to_cloud(subject, nwbFullpath)

if REGISTER_WITH_DATAJOINT:
    # NOTE: djsumit is imported here to avoid trying to connect to DataJoint before this.
    from uobrainflex.utils import djtools
    djBehaviorSession = djtools.submit_behavior_session(nwbFullpath)


