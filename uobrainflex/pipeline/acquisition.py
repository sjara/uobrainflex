"""
DataJoint schema for data acquisition sessions.
"""

import datajoint as dj
from . import subject
from . import experimenter

schema = dj.schema('acquisition')

@schema
class BehaviorSession(dj.Manual):
    definition = """
    behavior_session_id: varchar(255)
    ---
    -> subject.Subject
    behavior_session_start_time:          datetime
    behavior_session_type:                varchar(255)
    behavior_training_stage:              varchar(255)
    -> experimenter.Experimenter
    behavior_filename:                    varchar(255)
    behavior_version:                     varchar(255)
    licks_total=-1:                       int
    licks_left_ratio=NULL:                float
    choices_total=-1:                     int
    choices_left_stim_ratio=NULL:         float
    hits_total=-1:                        int
    hits_left_ratio=NULL:                 float
    hit_rate_left=NULL:                   float
    hit_rate_right=NULL:                  float
    behavior_entry_ts=CURRENT_TIMESTAMP:  timestamp
    """

'''    
@schema
class TestTable(dj.Manual):
    definition = """
    behavior_session_id: varchar(255)
    ---
    -> subject.Subject
    behavior_session_start_time:          datetime
    behavior_session_type:                varchar(255)
    behavior_training_stage:              varchar(255)
    -> experimenter.Experimenter
    behavior_filename:                    varchar(255)
    behavior_version:                     varchar(255)
    licks_total=-1:                       int
    licks_left_ratio=-1:                  float
    choices_total=-1:                     int
    choices_left_ratio=-1:                float
    hits_total=-1:                        int
    hits_left_ratio=-1:                   float
    performance=NULL:                     float
    behavior_entry_ts=CURRENT_TIMESTAMP:  timestamp
    """
'''    

'''
@schema
class TestTable(dj.Manual):
    definition = """
    test_table_id:  varchar(255)
    ---
    -> experimenter.Experimenter
    perform=NULL:      int
    sometext:       varchar(255)
    """
#    -> experimenter.Experimenter
'''


'''
@schema
class BehaviorSessionType(dj.Lookup):
    definition = """
    behavior_session_type: varchar(255)
    """
    #contents = [['attend auditory'],['attend visual'],['auditory'],['visual']]

@schema
class BehaviorTrainingStage(dj.Lookup):
    definition = """
    behavior_training_stage: varchar(255)
    """
    #contents = [['water on lick'],['discrimination']]

@schema
class BehaviorSession(dj.Manual):
    definition = """
    behavior_session_id: varchar(255)
    ---
    -> subject.Subject
    behavior_session_start_time:    datetime
    -> BehaviorSessionType
    -> BehaviorTrainingStage
    -> experimenter.Experimenter
    behavior_filename: varchar(255)
    behavior_version: varchar(255)
    behavior_entry_ts=CURRENT_TIMESTAMP:   timestamp
    """
'''
