"""
Definition of NWB file structure for behavior data from basic task.
"""

schema = \
{
    "session_description": "Behavior data. Two-alternative choice visual/auditory discrimination.",
    "institution": "University of Oregon",
    "species": "Mus musculus",
    "subject": {
        "filename": "Str_Animal_ID.txt"
    },
    "session_start_time": {
        "filename": "Date_Stimulation_Init.txt"
    },
    "experimenter": {
        "filename": "Str_Experimenter.txt"
    },
    "metadata": {
        "rig": {
            "filename": "Str_Rig_ID.txt"
        },
        "session_type": {
            "filename": "Str_Session_Type.txt"
        },
        "training_stage": {
            "filename": "Str_Training_Stage.txt"
        },
        "behavior_version": {
            "filename": "Str_Vehavior_Version.txt"
        },
    },
    "acquisition": {
        "pupil_size": {
            "description": "Area of pupil.",
            "filename": "pupil_size.txt",
            "timestamps": "pupil_time.txt",
            "unit": ""
        },
        "running_speed": {
            "description": "Speed of the animal, estimated from the rotary encoder on the wheel.",
            "filename": "Num_Encoder_Value.txt",
            "timestamps": "Time_Encoder_Time.txt",
            "unit": "cm/s"
        },
	"licks_left": {
            "description": "Time of each left-port lick.",
            "timestamps": "Time_Left_Lick.txt",
            "unit": "s"
        },
        "licks_right": {
            "description": "Time of each right-port lick.",
            "timestamps": "Time_Right_Lick.txt",
            "unit": "s"
        },
	"reward_left": {
            "description": "Time of each left-port reward.",
            "timestamps": "Time_Left_Pump.txt",
            "unit": "s"
        },
	"reward_right": {
            "description": "Time of each right-port reward.",
            "timestamps": "Time_Right_Pump.txt",
            "unit": "s"
        }
        
    },
    "trials": {
        "start_time": {
            "description": "Start time of trial",
            "filename": "Time_Trial_Start.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"target_modality": {
            "description": "Target sensory modality",
            "map": {'auditory':0, 'visual':1},
            "filename": "Num_Target_Modality.txt",
            "dtype": "int"
        },
	"cue_time": {
            "description": "Odor cue time rlative to session start",
            "filename": "Time_Trial_Start.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"cue_ID": {
            "description": "Odor identity",
            "map": {'A':0, 'B':1},
            "filename": "Num_Odor_ID.txt",
            "dtype": "int"
        },
	"cue_duration": {
            "description": "Duration of odor presentation",
            "filename": "Num_Odor_Dur.txt",
            "dtype": "float",
	    "unit": "milliseconds"
        },
	"stimulus_type": {
            "description": "Type of stimulus presented",
            "map": {'target':0, 'distractor':1, 'both':2},
            "filename": "Num_Trial_Type.txt",
            "dtype": "int"
        },
	"stimulus_time": {
            "description": "Stimulus time rlative to session start",
            "filename": "Time_Stimulus_Time.txt",
            "dtype": "float",
	    "unit": "s"
        },
	"target_port": {
            "description": "Port rewarded on correct response.",
            "map": {'left':0, 'right':1},
            "filename": "Num_Target_Port.txt",
            "dtype": "int"
        },
	"auditory_stim_id": {
            "description": "Auditory stimulus ID",
            "map": {'left':0, 'right':1, 'no stim':2},
	    "filename": "Num_Auditory_ID.txt",
            "dtype": "int"
        },
	"auditory_stim_difficulty": {
            "description": "Tone cloud coherence",
	    "filename": "Num_Tone_Cloud_Coherence.txt",
            "dtype": "float"
        },
	"visuam_stim_id": {
            "description": "Visual stimulus ID",
            "map": {'left':0, 'right':1, 'no stim':2},
	    "filename": "Num_Visual_ID.txt",
            "dtype": "int"
        },
	"outcome": {
            "description": "Trial outcome",
            "map": {'timeout':0, 'hit':1, 'miss':2, 'false_alarm':4, 'correct_reject':8, 'incorrect_reject':16},
	    "filename": "Num_Trial_Outcome.txt",
            "dtype": "int"
        },
	"stop_time": {
            "description": "Endtime of trial",
            "filename": "Time_Trial_End.txt",
            "dtype": "float",
	    "unit": "s"
        },
    }
}


# --- To fix: ---
'''
	"visual_stim_difficulty": {
            "description": "TBD",
	    "filename": "Str_Visual_Stimulus.txt",
            "dtype": "str"
        },
'''
