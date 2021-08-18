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
	    "visual_stim_association": {
            "filename": "Visual_Stim_Map.txt"
        },
        "auditory_stim_association": {
            "filename": "Auditory_Stim_Map.txt"
        }
    },
    "acquisition": {
	"whisker_energy": {
            "description": "Pixel intensity time differential within whisker pad ROI.",
            "filename": "Whisker_Energy.txt",
            "timestamps": "LV_FrameClock.txt",
            "unit": "a.u."
        },
	"pupil_area": {
            "description": "The pixel area of the pupil per frame calculated online during behavioral performance.",
            "filename": "Pupil_Area.txt",
            "timestamps": "LV_FrameClock.txt",
            "unit": "sq_pixels"
        },
	"pupil_diameter": {
            "description": "Pixel length of the long axis of the pupil per frame calculated online during behavioral performance.",
            "filename": "Pupil_Diameter.txt",
            "timestamps": "LV_FrameClock.txt",
            "unit": "pixels"
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
    "auditory_stim_band": {
            "description": "Target band of presented tone cloud",
            "map": {'low_band':0, 'high_band':1},
 	    "filename": "Tone_Cloud_Octave.txt",
            "dtype": "int"
        },
	"auditory_stim_difficulty": {
            "description": "Ratio of tones selected from target frequency band",
	    "filename": "Tone_Cloud_Coherence.txt",
            "dtype": "float"
        },
	"visual_stim_id": {
            "description": "Visual stimulus ID",
            "map": {'left':0, 'right':1, 'no stim':2},
	    "filename": "Visual_ID.txt",
            "dtype": "int"
        },
    "visual_stim_oreintation": {
            "description": "Orientation of gabor patch",
            "map": {'horizontal':0, 'vertical':1},
 	    "filename": "Visual_Stim_Orientation.txt",
            "dtype": "int"
        },
	"visual_stim_difficulty": {
            "description": "Delta degrees clockwise or counterclockwise from 45 degrees of drifting grating",
            "map": {'45':0, '36': 1, '27':2, '18':3, '9':4},
	    "filename": "Visual_Stim_Difficulty.txt",
            "dtype": "int"
        },
	"outcome": {
            "description": "Trial outcome",
            "map": {'timeout':0, 'hit':1, 'miss':2, 'false_alarm':4, 'correct_reject':8, 'incorrect_reject':16},
	    "filename": "Trial_Outcome.txt",
            "dtype": "int"
        },
	"choice": {
            "description": "Trial choice",
            "map": {'left':0, 'right':1, 'no_lick':2},
	    "filename": "Choice.txt",
            "dtype": "int"
        },
	"stop_time": {
            "description": "Endtime of trial",
            "filename": "Trial_End.txt",
            "dtype": "float",
	    "unit": "s"
        }
    }
}