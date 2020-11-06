"""
Connect to DataJoint database.
"""

import datajoint as dj
from .. import config

dj.config['database.host'] = config['DATAJOINT']['database_host']
dj.config['database.user'] = config['DATAJOINT']['database_user']
if 'database_pass' in config['DATAJOINT']:
    dj.config['database.password'] = config['DATAJOINT']['database_pass']
dj.conn()
