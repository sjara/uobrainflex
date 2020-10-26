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
        "filename": "Date_Stimulation_Init.txt"
    },
    "experimenter": {
        "filename": "experimenter.txt"
    },
    "acquisition": {
        "pupil_size": {
            "description": "Area of pupil.",
            "filename": "pupil_size.txt",
            "timestamps": "pupil_time.txt",
            "unit": ""
        },
        "licks_left": {
            "description": "Time of each left-port lick.",
            "timestamps": "Date_Left Lick.txt",
            "unit": ""
        },
        "licks_right": {
            "description": "Time of each right-port lick.",
            "timestamps": "Date_Right Lick.txt",
            "unit": ""
        },
        "running_speed": {
            "description": "Speed of the animal, estimated from the rotary encoder on the wheel.",
            "filename": "Encoder_Value.txt",
            "timestamps": "Date_Encoder_Time.txt",
            "unit": "cm/s"
        }
    },
    "trials": {
        "auditory_stim_id": {
            "description": "Auditory stimulus ID",
            "filename": "Auditory ID.txt",
            "dtype": "int"
        },
        "tone_cloud_coherence": {
            "description": "Coherence of the auditory tone-cloud stimulus.",
            "filename": "Tone Cloud Coherence.txt",
            "dtype": "float"
        },
        "target_port": {
            "description": "Port rewarded on correct response.",
            "map": {'left':0, 'right':1},
            "filename": "Target Port.txt",
            "dtype": "int"
        }
    }
}
