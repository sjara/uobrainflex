"""
Package for project "Brain States and Flexible Behavior".
"""

import os
import sys
import pathlib
import configparser

# -- Read configuration file --
_packageDir = os.path.dirname(os.path.abspath(__file__))
_configDir = os.path.split(_packageDir)[0] # One directory above
_configBasename = 'uobrainflex.conf'

configFile = os.path.join(_configDir,_configBasename)
if not os.path.isfile(configFile):
    msg = 'Configuration file not found: {}'
    raise IOError(msg.format(os.path.abspath(configFile)))
config = configparser.ConfigParser()
config.read(configFile)
