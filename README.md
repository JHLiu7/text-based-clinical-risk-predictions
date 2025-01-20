# Text-Based Clinical Risk Prediction

## Overview

This repository contains code and resources for predicting clinical risks using text-based data. The goal is to leverage natural language processing (NLP) techniques to analyze clinical notes to predict potential risks.

## Features

- Preprocessing of clinical text data
- Prompting LLMs for risk prediction
- Implementation of an novel hierachical network to handle various note categories


## In-hospital Mortality Prediction

1. **Patient Cohort**: We prepare the test cohort for in-hospital mortality prediction in `hosp_mort/test-cohort-3386.csv` for 3386 patients from MIMIC-III. Each patient has a list of note ids (`ROW_ID` in the note table) that are charted by the first day of ICU admission. Empty notes and duplicated notes charted on the same time were removed.
2. **Data Preprocessing**: Prepare your clinical text data using the provided preprocessing script `prepare_data.py`, which requries the csv file from Step 1 and `NOTEEVENTS.csv.gz` from MIMIC-III. Modify the path variables with the script accordingly.
3. **Risk Prediction with LLMs**: We provide a script to prompt LLMs to produce a numerical risk score for patients based on their clinical notes. Please see `hosp_mort/prompt_llm.py`, which has been examined on 34 LLMs listed in `list_of_llm.sh` and five instructions.
4. **Note-Specific Hierarchical Network**: We provide a reference implementation of NSHNet, a modular network to handle clinical notes of different categories with dedicated modules.



