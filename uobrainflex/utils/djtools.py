"""
Submit metadata to DataJoint.
"""

import os
import pynwb
from . import djconnect
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema
from uobrainflex.pipeline import experimenter as experimenterSchema
import warnings

# -- Ignore warning from NWB about namespaces already loaded --
warnings.filterwarnings("ignore", message="ignoring namespace '.*' because it already exists")


def submit_behavior_session(nwbFullpath):
    """
    Add behavior session metadata to DataJoint database.

    Args:
        nwbFullpath (str): full path to NWB data file to register with the DJ.

    Returns:
        djBehaviorSession: DataJoint object pointing to the table of behavior sessions.
    """
    
    ioObj = pynwb.NWBHDF5IO(nwbFullpath, 'r', load_namespaces=True)
    nwbFileObj = ioObj.read()
    subject = nwbFileObj.subject.subject_id
    experimenter = nwbFileObj.experimenter[0] # Use only first experimenter
    startTime =  nwbFileObj.session_start_time
    sessionType = nwbFileObj.lab_meta_data['session_metadata'].session_type
    trainingStage = nwbFileObj.lab_meta_data['session_metadata'].training_stage
    behaviorVersion = nwbFileObj.lab_meta_data['session_metadata'].behavior_version
    ioObj.close()
    filename = os.path.basename(nwbFullpath)
    
    djSubject = subjectSchema.Subject()
    djExperimenter = experimenterSchema.Experimenter()
    djSessionType = acquisitionSchema.BehaviorSessionType()
    djTrainingStage = acquisitionSchema.BehaviorTrainingStage()
    djBehaviorSession = acquisitionSchema.BehaviorSession()

    if subject not in djSubject.fetch('subject_id'):
        errMsg = 'You need to add new subjects manually before adding sessions.'
        raise Exception(errMsg)

    if experimenter not in djExperimenter.fetch('experimenter_id'):
        errMsg = 'You need to add new experimenters manually before adding sessions.'
        raise Exception(errMsg)

    djSessionType.insert1([sessionType], skip_duplicates=True)
    djTrainingStage.insert1([trainingStage], skip_duplicates=True)

    # See pipeline.acquisition for order of attributes
    newSession = (subject, startTime, sessionType, trainingStage,
                  experimenter, filename, behaviorVersion, None)

    # FIXME: currently it defaults to replace
    djBehaviorSession.insert1(newSession, replace=True)
    print('Inserted new session {}'.format(newSession))

    return djBehaviorSession
