"""
Read metadata from NWB file created by uobrainflex.
"""

import os
import pynwb
from uobrainflex import config

subject = 'BW020'
session = '20210409T100132'

dataDir = config['BEHAVIOR']['nwb_data_path']
filename = '{}_behavior_{}.nwb'.format(subject, session)
nwbFullpath = os.path.join(dataDir, subject, filename)

ioObj = pynwb.NWBHDF5IO(nwbFullpath, 'r', load_namespaces=True)
nwbFileObj = ioObj.read()

# NOTE: if you want to avoid the warning when loading the file use code below
# import warnings
# msg = "ignoring namespace '.*' because it already exists"
# warnings.filterwarnings("ignore", message=msg)

metadata = nwbFileObj.lab_meta_data['metadata']

print('You access the metadata as object attributes:')
print('metadata.rig : {}'.format(metadata.rig))
print('or access them as a dictionary:')
print('metadata.fields : {}'.format(metadata.fields))

ioObj.close()
