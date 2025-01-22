# SIGIR2025_LLMAdvisor_supplementary

# Data Analysis for User Study

This repository contains scripts for analyzing data from a user study.

---

## Setup
1. Install the required packages:

   `pip install -r requirements.txt`

2. Place the necessary data files in the `study_data` directory.

---

## Script Descriptions

### prepare_dataset.py
This script performs data preprocessing.

- `filter_dataframes_on_uuid(*dataframes, valid_uuids)`: Filters multiple dataframes by `user_id_uuid`.
- `filter_out_invalid_data(interaction, summarization, round_evaluation, final_ranking, final_survey)`: Filters out invalid data.

### stats_analysis.py
This script performs statistical analysis.

- `final_perception_statistical_analysis(pair_df, final_survey, control_experiment_id)`: Performs statistical analysis on the final evaluation.
- `main()`: Loads and preprocesses data, then performs statistical analysis.

---

## Usage

1. **Run the statistical analysis**:

   `python analysis_script/stats_analysis.py`

---

## License
This project is licensed under the MIT License.

