"""
Definition of NWB file structure for behavior data from basic task.
"""

nwbformat = \
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
            "filename": "Encoder_Value.txt",
            "timestamps": "x"
        }
    },
    "processing": {
        "pupil_size": {
            "filename": "",
            "timestamps": ""
        }
    },
    "trials": {
        "auditory_stim_id": {
            "description": "Auditory stimulus ID",
            "filename": "Auditory ID.txt",
            "dtype": "int"
        },
        "sound_frequency": {
            "description": "Sound frequency",
            "unit": "Hz",
            "filename": "Hz.txt",
            "dtype": "float"
        },
        "target_port": {
            "description": "Port rewarded on correct response",
            "filename": "Target Port.txt",
            "dtype": "int"
        }
    }
}
