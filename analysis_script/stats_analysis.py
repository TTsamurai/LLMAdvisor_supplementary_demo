import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, wilcoxon, ttest_ind
from scipy.stats import kendalltau, spearmanr
from wordcloud import WordCloud

# from huggingface_hub import login     # (Only import if you really need this)
from prepare_dataset import (
    filter_data_by_experiment_id,
    get_context_list,
    get_company_label,
)

# Global Constants
FINAL_EVAL_COLUMNS = [
    "perceived_personalization",
    "emotional_trust",
    "trust_in_competence",
    "intention_to_use",
    "usefulness",
    "overall_satisfaction",
    "providing_information",
]
ROUND_EVAL_COLS = ["likelihood", "confidence", "familiarity"]

# ------------------------------------------------
# 1. PLOTTING FUNCTIONS
# ------------------------------------------------


def plot_conv_length(df, output_dir, exp_id):
    """Plot a histogram of conversation length (number of user messages)."""
    sizes = df.query("role == 'user'").groupby("user_id_uuid").size()
    median_value = sizes.median()

    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=30, color="blue", alpha=0.7)
    plt.axvline(median_value, color="red", linestyle="dashed", linewidth=1)
    plt.text(
        median_value, plt.ylim()[1] * 0.9, f"Median: {median_value:.2f}", color="red"
    )
    plt.xlabel("Conversation Length")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Conversation Length in Experiment {exp_id}")
    plt.savefig(os.path.join(output_dir, f"conv_length_{exp_id}.png"))
    plt.close()


def process_group(df):
    """
    Given a group of timestamps, compute the time differences for only user responses.
      - Drop the first timestamp (not meaningful for a difference).
      - Then compute consecutive differences.
      - Keep every other difference (user-labeled rows), starting from index 1.
    """
    df = df.iloc[1:]  # Drop the first row
    df = df.diff()  # Compute consecutive diffs
    df = df.iloc[1::2]  # Keep every other row (1, 3, 5, ...)
    return df


def plot_response_time(df, output_dir, exp_id):
    """
    Plot distribution of response times (in seconds), ignoring:
      - NaN values
      - Values >= 300 seconds
    """
    response_time = (
        df.groupby(["user_id_uuid", "comp_type"])["timestamp"]
        .apply(process_group)
        .dt.total_seconds()
        .dropna()
    )
    original_count = len(response_time)
    response_time = response_time[response_time < 300]  # discard outliers
    removed_count = original_count - len(response_time)
    print(f"Removed {removed_count} invalid response times (>=300s)")

    median_value = response_time.median()
    plt.figure(figsize=(10, 6))
    plt.hist(response_time, bins=100, color="blue", alpha=0.7)
    plt.axvline(median_value, color="red", linestyle="dashed", linewidth=1)
    plt.text(
        median_value, plt.ylim()[1] * 0.9, f"Median: {median_value:.2f}", color="red"
    )
    plt.xlabel("Response Time (seconds)")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.title(f"Histogram of Response Time in Experiment {exp_id}")
    plt.savefig(os.path.join(output_dir, f"response_time_{exp_id}.png"))
    plt.close()


def plot_user_perception(df, output_dir, exp_id):
    """Plots histograms for each of the final evaluation columns."""
    for col in FINAL_EVAL_COLUMNS:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=5, color="blue", alpha=0.7)
        mean_value = df[col].mean()
        plt.axvline(mean_value, color="red", linestyle="dashed", linewidth=1)
        plt.text(
            mean_value, plt.ylim()[1] * 0.9, f"Mean: {mean_value:.2f}", color="red"
        )
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col} in Experiment {exp_id}")
        plt.savefig(os.path.join(output_dir, f"{col}_{exp_id}.png"))
        plt.close()


def plot_reason_word_cloud(text, output_dir, name="reason_wordcloud"):
    """Generate and save a word cloud based on a given text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()


def plot_user_elicitation_correct(user_elicitation_correct, output_dir, exp_id):
    """
    Plots a histogram showing how many users were 'correct' in their elicitation
    (usually 'correct' is a binary or integer-coded value).
    """
    plt.figure(figsize=(10, 6))
    user_elicitation_correct["correct"].hist()
    plt.xlabel("Correct")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Correct in Narrative {exp_id}")
    plt.savefig(os.path.join(output_dir, f"correct_{exp_id}.png"))
    plt.close()


# ------------------------------------------------
# 2. STATISTICAL TEST FUNCTIONS
# ------------------------------------------------


def final_perception_statistical_analysis(pair_df, final_survey, control_experiment_id):
    """
    1. Splits data into Control and Treatment groups based on `control_experiment_id`.
    2. Checks normality using Shapiro-Wilk.
    3. Performs a Wilcoxon signed-rank test on matched pairs of columns.
    """
    # Separate user-ids for control vs. treatment
    user_id_uuid_with_control = []
    user_id_uuid_without_treatment = []

    for _, row in pair_df.iterrows():
        if row["first_user_experiment_id"] == control_experiment_id:
            user_id_uuid_with_control.append(row["first_user_id_uuid"])
            user_id_uuid_without_treatment.append(row["second_user_id_uuid"])
        else:
            user_id_uuid_with_control.append(row["second_user_id_uuid"])
            user_id_uuid_without_treatment.append(row["first_user_id_uuid"])

    # Filter and realign final survey data
    final_survey_control = final_survey[
        final_survey["user_id_uuid"].isin(user_id_uuid_with_control)
    ]
    final_survey_control = final_survey_control.set_index("user_id_uuid").loc[
        user_id_uuid_with_control
    ]

    final_survey_treatment = final_survey[
        final_survey["user_id_uuid"].isin(user_id_uuid_without_treatment)
    ]
    final_survey_treatment = final_survey_treatment.set_index("user_id_uuid").loc[
        user_id_uuid_without_treatment
    ]

    # Collect columns as lists for normality checks
    rq1_final_survey = {
        col: final_survey_control[col].dropna().tolist() for col in FINAL_EVAL_COLUMNS
    }
    rq2_final_survey = {
        col: final_survey_treatment[col].dropna().tolist() for col in FINAL_EVAL_COLUMNS
    }

    print("[Normality Check] Shapiro-Wilk Test (p < 0.05 -> not normal)")
    print("Control Group:")
    for col, values in rq1_final_survey.items():
        pval = shapiro(values).pvalue if len(values) >= 3 else np.nan
        print(f"  {col}: p={pval:.3f} (n={len(values)})")

    print("Treatment Group:")
    for col, values in rq2_final_survey.items():
        pval = shapiro(values).pvalue if len(values) >= 3 else np.nan
        print(f"  {col}: p={pval:.3f} (n={len(values)})")

    # Wilcoxon signed-rank test requires matched pairs
    print("[Wilcoxon Signed-Rank Test]")
    wilcoxon_test_result = {}
    for col in FINAL_EVAL_COLUMNS:
        data_rq1 = rq1_final_survey[col]
        data_rq2 = rq2_final_survey[col]

        # Ensure both lists have the same length and are not empty
        if len(data_rq1) == len(data_rq2) and len(data_rq1) > 0:
            stat, p_value = wilcoxon(data_rq1, data_rq2)
            wilcoxon_test_result[col] = f"{p_value:.3f}"
        else:
            wilcoxon_test_result[col] = "Data mismatch or not available"

    return wilcoxon_test_result


# ------------------------------------------------
# 3. MAIN ANALYSIS
# ------------------------------------------------


def main():
    # --- 3.1 Load and prepare data ---
    external_data_path = "../study_data/experiment_processed_data.jsonl"
    context_info_list = get_context_list(external_data_path)
    company_level_label = get_company_label(context_info_list)

    final_ranking = pd.read_excel("../study_data/user_study_data/final_ranking.xlsx")
    interaction = pd.read_csv("../study_data/user_study_data/advisory_discussion.csv")
    final_survey = pd.read_csv("../study_data/user_study_data/advisor_assessment.csv")
    summarization = pd.read_csv(
        "../study_data/user_study_data/user_profile_and_annotation.csv"
    )
    user_elicitation_manual_analysis = pd.read_csv(
        "../study_data/user_study_data/user_profile_and_annotation.csv"
    )

    # Convert stringified lists to actual lists
    final_ranking["sorted_user_ranking"] = final_ranking["sorted_user_ranking"].apply(
        ast.literal_eval
    )
    final_ranking["sorted_gold_ranking"] = final_ranking["sorted_gold_ranking"].apply(
        ast.literal_eval
    )

    # Add experiment_id and success columns
    user_elicitation_manual_analysis["experiment_id"] = (
        user_elicitation_manual_analysis["user_id"].apply(lambda x: x.split("_")[1])
    )
    user_elicitation_manual_analysis["success"] = (
        user_elicitation_manual_analysis["correct"] >= 2
    ).astype(int)

    # Merge with manual analysis
    final_ranking_with_manual = pd.merge(
        final_ranking, user_elicitation_manual_analysis, on="user_id_uuid", how="inner"
    )
    final_ranking_with_manual["experiment_id"] = final_ranking_with_manual[
        "user_id_uuid"
    ].apply(lambda x: x.split("_")[1])

    # Compute correlation scores (Kendall, Spearman)
    final_ranking["kendall"] = final_ranking.apply(
        lambda x: kendalltau(
            x["sorted_user_ranking"], x["sorted_gold_ranking"]
        ).statistic,
        axis=1,
    )
    final_ranking["spearmans"] = final_ranking.apply(
        lambda x: spearmanr(
            x["sorted_user_ranking"], x["sorted_gold_ranking"]
        ).correlation,
        axis=1,
    )

    final_ranking_with_manual["spearmans"] = final_ranking_with_manual.apply(
        lambda x: spearmanr(
            x["sorted_user_ranking"], x["sorted_gold_ranking"]
        ).correlation,
        axis=1,
    )

    # --- 3.2 Example: T-tests for success vs. non-success users in each experiment ---
    def get_spearmans_by_exp(df, exp_id, success=None):
        """Return spearman correlation list filtered by experiment and optionally by success."""
        query_str = f"experiment_id == '{exp_id}'"
        if success is not None:
            query_str += f" and success == {int(success)}"
        return df.query(query_str)["spearmans"].dropna().tolist()

    # Example usage with experiment 0, 2, 3
    exp_0_success = get_spearmans_by_exp(final_ranking_with_manual, 0, success=True)
    exp_0_non_success = get_spearmans_by_exp(
        final_ranking_with_manual, 0, success=False
    )
    t_stat_0, p_value_0 = ttest_ind(exp_0_non_success, exp_0_success, equal_var=False)
    print(f"Exp 0 success vs. non-success T-test: p={p_value_0:.3f}")

    # Repeat for other experiments as needed...
    exp_2_success = get_spearmans_by_exp(final_ranking_with_manual, 2, success=True)
    exp_2_non_success = get_spearmans_by_exp(
        final_ranking_with_manual, 2, success=False
    )
    t_stat_2, p_value_2 = ttest_ind(exp_2_non_success, exp_2_success, equal_var=False)
    print(f"Exp 2 success vs. non-success T-test: p={p_value_2:.3f}")

    # etc...

    # --- 3.3 Paired user analysis for personalization, personality, etc. ---
    pair = pd.read_csv("../study_data/valid_user_id_uuid.csv").dropna()
    pair["first_user_experiment_id"] = pair["first_user_id_uuid"].apply(
        lambda x: x.split("_")[1]
    )
    pair["second_user_experiment_id"] = pair["second_user_id_uuid"].apply(
        lambda x: x.split("_")[1]
    )

    final_survey["user_id_uuid"] = final_survey["user_id"] + "_" + final_survey["uuid"]
    final_survey["experiment_id"] = final_survey["user_id"].apply(
        lambda x: x.split("_")[1]
    )

    # Personalization: compare exp_id 0 vs. 1
    pair_personalization = pair.query(
        "first_user_experiment_id in ['0','1']"
    ).reset_index(drop=True)
    wilcoxon_personalization = final_perception_statistical_analysis(
        pair_personalization, final_survey, control_experiment_id="0"
    )
    print("Wilcoxon - Personalization (0 vs. 1):")
    print(wilcoxon_personalization)

    # Personality: compare exp_id 2 vs. 3
    pair_personality = pair.query("first_user_experiment_id in ['2','3']").reset_index(
        drop=True
    )
    wilcoxon_personality = final_perception_statistical_analysis(
        pair_personality, final_survey, control_experiment_id="2"
    )
    print("Wilcoxon - Personality (2 vs. 3):")
    print(wilcoxon_personality)

    # ------------------------------------------------
    # Additional analysis, plots, etc. as needed...
    # ------------------------------------------------


if __name__ == "__main__":
    main()
