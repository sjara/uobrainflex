This folder contains code associated with Hulsey et al (2023) *"Decision-making dynamics are predicted by arousal and uninstructed movements"*.

The `data_processing_code` folder contains scripts to:
1. Gather subject data from NWB files.
2. Perform model selection to determine appropriate number of glm-hmm states.
3. Fit a final glm-hmm to subject data. 
4. Determine optimal pupil diameters.
5. Perform state classification with arousal and movement measures.
6. Calculate predictive accuracies of models with appropriate number of HMM states.
7. Calculate predictive accuracies for models with up to 7 GLM-HMM states for example subject.


The `figure_generation_code` folder contains scripts to generate paper figures from intermediate data saved by the processing scripts.

The scripts run using the uobrainflex environment (https://github.com/sjara/uobrainflex), and a fork of the lindermanlab ssm package (https://github.com/mazzulab/ssm).
Alternatively, the only alterations to the linerman lab ssm package (https://github.com/lindermanlab/ssm) required are included in the 'ssm' folder.

NWB data files have been deposited on DANDI and are publicly available as of the date of publication. Accession numbers are listed in the key resources table of the publication.	

