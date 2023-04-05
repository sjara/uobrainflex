This folder contains code associated with Hulsey et al (2023) *"Decision-making dynamics are predicted by arousal and uninstructed movements"*.

The `data_processing_code` folder contains scripts to:
1. Gather subject data from NWB files.
2. Perform model selection to determine appropriate number of glm-hmm states.
3. Fit a final glm-hmm to subject data. 
4. Determine optimal pupil diameters.
5. Perform state classification with arousal and movement measures.

The `figure_generation_code` folder (in progress) contains scripts to generate paper figures from intermediate data saved by the processing scripts.

The scripts run using the uobrainflex environment (https://github.com/sjara/uobrainflex), and a fork of the lindermanlab ssm package (https://github.com/mazzulab/ssm).

NWB data files will be available upon request.

