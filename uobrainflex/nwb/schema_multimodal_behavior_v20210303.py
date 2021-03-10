"""
Definition of NWB file structure for behavior data from basic task.
"""

schema = \
{
    "session_description": "Behavior data. Two-alternative choice visual/auditory discrimination.",
    "institution": "University of Oregon",
    "species": "Mus musculus",
    "subject": {
        "filename": "Animal_ID.txt"
    },
    "session_start_time": {
        "filename": "Session_Start_Datetime.txt"
    },
    "experimenter": {
        "filename": "Experimenter.txt"
    },
    "metadata": {
        "rig": {
            "filename": "Rig_ID.txt"
        },
        "session_type": {
            "filename": "Session_Target_Modality.txt"
        },
        "training_stage": {
            "filename": "Training_Stage.txt"
        },
        "behavior_version": {
            "filename": "Behavior_Version.txt"
        },
    },
    "acquisition": {
        "pupil_size": {
            "description": "Area of pupil.",
            "filename": "Pupil_Size.txt",
            "timestamps": "Pupil_Time.txt",
            "unit": ""
        },
	"whisker_energy": {
            "description": "Pixel intensity time differential within whisker pad ROI.",
            "filename": "Whisker_Energy.txt",
            "timestamps": "Pupil_Time.txt",
            "unit": ""
        },
        "running_speed": {
            "description": "Speed of the animal, estimated from the rotary encoder on the wheel.",
            "filename": "Encoder_Value.txt",
            "timestamps": "Encoder_Time.txt",
            "unit": "cm/s"
        },
	"licks_left": {
            "description": "Time of each left-port lick.",
            "timestamps": "Left_Lick.txt",
            "unit": "s"
        },
        "licks_right": {
            "description": "Time of each right-port lick.",
            "timestamps": "Right_Lick.txt",
            "unit": "s"
        },
	"reward_left": {
            "description": "Time of each left-port reward.",
            "timestamps": "Left_Pump.txt",
            "unit": "s"
        },
	"reward_right": {
            "description": "Time of each right-port reward.",
            "timestamps": "Right_Pump.txt",
            "unit": "s"
        }
        
    },
    "trials": {
        "start_time": {
            "description": "Start time of trial",
            "filename": "Trial_Start.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"target_modality": {
            "description": "Target sensory modality",
            "map": {'auditory':0, 'visual':1},
            "filename": "Target_Modality.txt",
            "dtype": "int"
        },
	"cue_time": {
            "description": "Odor cue time relative to session start",
            "filename": "Trial_Start.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"cue_ID": {
            "description": "Odor identity",
            "map": {'A':0, 'B':1},
            "filename": "Odor_ID.txt",
            "dtype": "int"
        },
	"cue_duration": {
            "description": "Duration of odor presentation",
            "filename": "Odor_Dur.txt",
            "dtype": "float",
	    "unit": "milliseconds"
        },
	"stimulus_type": {
            "description": "Type of stimulus presented",
            "map": {'target':0, 'distractor':1, 'both':2},
            "filename": "Trial_Type.txt",
            "dtype": "int"
        },
	"stimulus_time": {
            "description": "Stimulus time rlative to session start",
            "filename": "Stimulus_Time.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"stimulus_duration": {
            "description": "Duration of stimulus presentation",
            "filename": "Stimulus_Dur.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"target_port": {
            "description": "Port rewarded on correct response",
            "map": {'left':0, 'right':1},
            "filename": "Target_Port.txt",
            "dtype": "int"
        },
	"auditory_stim_id": {
            "description": "Auditory stimulus ID",
            "map": {'left':0, 'right':1, 'no stim':2},
	    "filename": "Auditory_ID.txt",
            "dtype": "int"
        },
	"auditory_stim_difficulty": {
            "description": "Tone cloud coherence",
	    "filename": "Tone_Cloud_Coherence.txt",
            "dtype": "float"
        },
	"visual_stim_id": {
            "description": "Visual stimulus ID",
            "map": {'left':0, 'right':1, 'no stim':2},
	    "filename": "Visual_ID.txt",
            "dtype": "int"
        },
	"outcome": {
            "description": "Trial outcome",
            "map": {'timeout':0, 'hit':1, 'miss':2, 'false_alarm':4, 'correct_reject':8, 'incorrect_reject':16},
	    "filename": "Trial_Outcome.txt",
            "dtype": "int"
        },
	"stop_time": {
            "description": "Endtime of trial",
            "filename": "Trial_End.txt",
            "dtype": "float",
	    "unit": "s"
        },
    }
}


# --- To fix: ---
'''
	"visual_stim_difficulty": {
            "description": "TBD",
	    "filename": "Visual_Stimulus.txt",
            "dtype": "str"
        },
'''
