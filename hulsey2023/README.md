This folder contains code associated with Hulsey et al (2023) "Transitions between discrete performance states in auditory and visual tasks are predicted by arousal and uninstructed movements".

data_processing_code folder contains scripts to 1. gather subject data from nwb files 2. perform model selection to determine appropriate number of glm-hmm states 3. fit a final glm-hmm to subject data. 4. determine optimal pupil diameters 5. perform state classification with arousal and movement measures.
figure_generation_code folder (in process) contains scripts to generate paper figures from intermediate data saved by the processing scripts.

The scripts run using the uobrainfelx environment (https://github.com/sjara/uobrainflex), and a fork of the lindermanlab ssm package (https://github.com/mazzulab/ssm).
nwb data files will be available upon request.

