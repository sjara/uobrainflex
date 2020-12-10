"""
Convert behavior data to NWB format, copy data to cloud storage,
and register the data onto a DataJoint database.

Many of these steps require your configuration file (uobrainflex.conf)
to indicate the correct paths for cloud storage and DataJoint.

Options for how to use this script:

1. Run it by specifying the file containing a list of directories to process:
   python convert_multiple_behavior.py /data/labview/list_of_folders.txt

2. Run it without arguments to open a GUI and select the file that specifies
   the list of directories to process:
   python convert_multiple_behavior.py

3. Run it by specifying a single directory to process:
   python convert_multiple_behavior.py /data/labview/testmouse_201028_141726/
"""

import os
import sys
import tkinter.filedialog
import tkinter
from uobrainflex import config
from uobrainflex.nwb import savebehavior
from uobrainflex.utils import cloudstorage
# NOTE: Modules related to DataJoint are imported below to avoid
#       trying to connect to the database unless it is necessary.
#
# import importlib
# importlib.reload(savebehavior)  # Useful during development


def read_list_of_folders(txtFile):
    '''Read list of folders to process from a txt file.'''
    with open(txtFile,'r') as foldersFile:
        return [line.strip() for line in foldersFile if line.strip()]
    
# -- Interpret command line arguments --
if len(sys.argv)>1:
    firstArg = sys.argv[1]
    if os.path.isdir(firstArg): # Process only the specified directory
        folders = [firstArg]
    elif os.path.isfile(firstArg): # Process all directories listed in the specified file
        folders = read_list_of_folders(firstArg)
    else:
        raise ValueError('The first argument needs to be an existing directory or file.')
else: # Process all directories listed in the file chosen via the GUI
    labview_data_path = config['BEHAVIOR']['labview_data_path']
    dialogTitle = 'Choose the txt file listing all directories to process'
    rootWin = tkinter.Tk()
    rootWin.withdraw()
    foldersFilename = tkinter.filedialog.askopenfilename(initialdir=labview_data_path,
                                                         title=dialogTitle)
    if not foldersFilename:
        raise ValueError('You need to select a valid file.')
    folders = read_list_of_folders(foldersFilename)

# -- Ask user if save and register --
saveToCloudStr = input('Copy data to cloud? [Y/n] ')
SAVE_TO_CLOUD = (saveToCloudStr.lower()=='y') or not saveToCloudStr

registerStr = input('Register session with DataJoint? [Y/n] ')
REGISTER_WITH_DATAJOINT = (registerStr.lower()=='y') or not registerStr

for inputDir in folders:
    print(f'\nProcessing {inputDir}')
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

    if CONVERT_TO_NWB:
        nwbFile, nwbPath = savebehavior.to_nwb(inputDir, subjectOutputDir)
        assert nwbPath==nwbFullpath  # Check that it saved in the correct location

    if SAVE_TO_CLOUD:
        cloudPath = cloudstorage.save_behavior_to_cloud(subject, nwbFullpath)

    if REGISTER_WITH_DATAJOINT:
        # NOTE: djsumit is imported here to avoid trying to connect to DataJoint before this.
        from uobrainflex.utils import djtools
        djBehaviorSession = djtools.submit_behavior_session(nwbFullpath)


