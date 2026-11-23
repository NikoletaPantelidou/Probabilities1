# Large Language Model probabilities cannot reliably distinguish between possible and impossible language

This repository contains code and data for the paper ["Large Language Model probabilities cannot reliably distinguish between possible and impossible language"]. The project is a clone from Hu et al., 2024 (see paper:(https://doi.org/10.1073/pnas.2400917121)). It has been minimally modified to preserve the methodology as much as possible. If you want to use this code, please cite the original repository by Hu et al.
In the current repository, we have added the task 'lmmg', which is related to probability extraction. We only use the material under the 'lmmg' folder.
If you are interested in the other tasks, please check the original repository by Hu et al. with the relevant README file.
## 0. Code structure

The relevant folders for the probability extraction are the following: 

- `configs`: YAML files used to control experiment settings, specifically the models configurations.
- `external_stimuli`: CSV files containing the stimuli.
- `output`: CSV files with outputs from model runs.
- `output_with_token_counts`: computes the token counts by model across the entire dataset.
- `src`: Python scripts for implementing the experiments.
- `metrics/lmmg` : compares the minimal pairs and computes the 'sum' probabilities.

Note: The folder 'figures' and 'analysis' are not used in this repository.

## Task & stimuli:
The task taps into minimal-pair surprisal differences, juxtaposing 50 grammatical sentences with (i) 50 lower frequency grammatical sentences, (ii) 50 ungrammatical sentences, (iii) 50 pragmatically odd sentences, and (iv) 50 semantically odd sentences.
In the folder, 'external_stimuli', within the 'lmmg' subfolder, you can find the full dataset in a file called 'stimuli.csv'.

## Models tested:
We elicit probabilities from 8 models that come from different families:
Gemma-2b 
Gemma-7b
Pythia-1b-deduped
Mistral-7b
DeepSeek-R1-Distill-Qwen-32b
Olmo-3.1-32b-Instruct
Qwen-2.5-32b-Instruct 
Qwen-2.5-7B-Instruct 

## Replication:
To replicate the experiment, you need:

- A computer/laptop with at least 475GB memory. 32B models will run slow.
- Set the Python interpreter to 3.14 version.
- Download all the necessary libraries and the requirements.txt.
- The file 'terminal.txt' contains all the relevant python instructions that can be run directly in the terminal.
