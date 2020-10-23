"""
Definition of NWB file structure for behavior data from basic task.
"""

schema = \
{
    "session_description": "Behavior data. Two-alternative choice visual/auditory discrimination.",
    "subject": {
        "filename": "subject.txt"
    },
    "session_start_time": {
        "filename": "session_start_time.txt"
    },
    "experimenter": {
        "filename": ""
    },
    "acquisition": {
        "rotary_encoder": {
            "description": "Value of rotary encoder on running wheel",
            "filename": "Encoder_Value.txt",
            "timestamps": "Date_Encoder_Time.txt",
            "unit": ""
        }
    },
    "processing": {
        '''
        "pupil_size": {
            "description": "Area of pupil",
            "filename": "",
            "timestamps": "",
            "unit": ""
        }
        '''
    },
    "trials": {
        "auditory_stim_id": {
            "description": "Auditory stimulus ID",
            "filename": "Auditory ID.txt",
            "dtype": "int"
        },
        "tone_cloud_coherence": {
            "description": "Coherence of the auditory tone-cloud stimulus",
            "filename": "Tone Cloud Coherence.txt",
            "dtype": "float"
        },
        "target_port": {
            "description": "Port rewarded on correct response",
            "filename": "Target Port.txt",
            "dtype": "int"
        }
    }
}
