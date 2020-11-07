"""
Utilities for cloud storage.
"""

import os
import shutil
from .. import config


def save_behavior_to_cloud(subject, filefullpath):
    """
    Copy a file to the cloud storage.

    Files are saved into a folder named the same as the subject and placed in the
    path defined in uobrainflex.conf (cloud_storage_path).

    Args:
        filefullpath (str): full path to file to transfer.

    Returns:
        cloudFullPath (str): full path to the file in cloud storage.
    """

    cloudDir = config['BEHAVIOR']['cloud_storage_path']
    # NOTE: the cloud storage needs to have the file 'verify_connection.empty'
    if not os.path.isfile(os.path.join(cloudDir,'verify_connection.empty')):
        raise IOError('Connection to cloud storage could not be established. '+
                      'Check you uobrainflex.conf')
    subjectCloudDir = os.path.join(cloudDir,subject)
    if not os.path.isdir(subjectCloudDir):
        msg = 'Created a new folder for this subject on cloud storage: {}'
        print(msg.format(subjectCloudDir))
        os.mkdir(subjectCloudDir)

    filenameOnly = os.path.basename(filefullpath)
    cloudFullPath = os.path.join(subjectCloudDir,filenameOnly)
    print('Copying NWB file to {} ...'.format(cloudFullPath))
    shutil.copy(filefullpath,cloudFullPath)
    print('Done copying file.')
    return(cloudFullPath)
