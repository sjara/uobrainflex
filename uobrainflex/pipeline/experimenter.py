"""
DataJoint schema for subject information.
"""

import datajoint as dj

schema = dj.schema('experimenter')

@schema
class Lab(dj.Lookup):
    definition = """
    lab:                 varchar(64)
    """
    contents = [['McCormick'],['Niell'],['Jaramillo'],
                ['Mazzucato'],['Smear'],['Wehr']]
    
@schema
class Experimenter(dj.Lookup):
    definition = """
    experimenter_id: varchar(32)
    ---
    experimenter_name: varchar(64)
    -> Lab
    experimenter_ts=CURRENT_TIMESTAMP: timestamp
    """



