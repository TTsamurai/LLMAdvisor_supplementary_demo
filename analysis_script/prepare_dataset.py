import os
from datasets import load_dataset_builder, load_dataset
from huggingface_hub import login
import pandas as pd
import json
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau, spearmanr
import ipdb


def filter_timestamp_for_experiment(df, sampling_period="2024-10-24"):
    date_threshold = pd.Timestamp(sampling_period)
    filtered_df = df[df["timestamp"] >= date_threshold]
    return filtered_df


def create_user_id_uuid(df):
    """Helper function to add 'user_id_uuid' to DataFrame"""
    if "user_id" in df.columns and "uuid" in df.columns:
        df["user_id_uuid"] = df["user_id"] + "_" + df["uuid"]
    else:
        raise ValueError("DataFrame must contain 'user_id' and 'uuid' columns")


def filter_dataframes_on_uuid(*dataframes, valid_uuids):
    """Filter multiple dataframes on the same 'user_id_uuid' criteria"""
    filtered_data = []
    for df in dataframes:
        if "user_id_uuid" in df.columns:
            df_filtered = (
                df[df["user_id_uuid"].isin(valid_uuids)].dropna().reset_index(drop=True)
            )
            filtered_data.append(df_filtered)
        else:
            raise ValueError("DataFrame does not contain 'user_id_uuid' column")
    return filtered_data


def filter_out_invalid_data(
    interaction, summarization, round_evaluation, final_ranking, final_survey
):
    # Create 'user_id_uuid' for all dataframes
    for df in (
        interaction,
        summarization,
        round_evaluation,
        final_ranking,
        final_survey,
    ):
        create_user_id_uuid(df)

    # Get set of valid 'user_id_uuid' from round_evaluation
    valid_uuids = set(final_ranking["user_id_uuid"])

    # Filter all dataframes based on valid 'user_id_uuid'
    interaction, summarization, round_evaluation, final_ranking, final_survey = (
        filter_dataframes_on_uuid(
            interaction,
            summarization,
            round_evaluation,
            final_ranking,
            final_survey,
            valid_uuids=valid_uuids,
        )
    )

    # Remove accidental data that I added
    rows_to_drop_interaction = interaction[
        (interaction["timestamp"].dt.date == pd.Timestamp("2024-11-11").date())
        & (interaction["user_id"] == "user_0_0_0")
    ]
    interaction = interaction.drop(rows_to_drop_interaction.index)
    rows_to_drop_summarization = summarization[
        (summarization["timestamp"].dt.date == pd.Timestamp("2024-11-11").date())
        & (summarization["user_id"] == "user_0_0_0")
    ]
    summarization = summarization.drop(rows_to_drop_summarization.index)

    # Return the filtered dataframes
    return interaction, summarization, round_evaluation, final_ranking, final_survey


def divide_interaction_by_type(interaction, interaction_type):
    assert interaction_type in ["financial_decision", "user_elicitation"]
    if interaction_type == "user_elicitation":
        return interaction[
            interaction["value"].apply(lambda x: x["type"]) == interaction_type
        ]
    else:
        return interaction[
            interaction["value"].apply(lambda x: x["type"]) != "user_elicitation"
        ]


def filter_data_by_experiment_id(df, experiment_id):
    experiment_id = str(experiment_id)
    assert experiment_id in ["0", "1", "2", "3"]

    return df[df.apply(lambda x: x["user_id"].split("_")[-3], axis=1) == experiment_id]


def filter_data_by_narrative_id(df, narrative_id):
    narrative_id = str(narrative_id)
    assert narrative_id in ["0", "1", "2", "3"]

    return df[df.apply(lambda x: x["user_id"].split("_")[-2], axis=1) == narrative_id]


def concat_conv_based_on_user_id(df):
    role_content_list = []
    for idx, row in df.iterrows():
        role_content_list.append(f'{row["value"]["role"]}: {row["value"]["content"]}')
    role_content_list = "\n".join(role_content_list)
    return role_content_list


def visualize_summarization_based_on_narrative_and_invester(df, output_dir):
    df.sort_values(by=["timestamp"], inplace=True)
    output_path = os.path.join(output_dir, "summarization")
    narratives = ["0", "1", "2"]
    for narrative in narratives:
        with open(f"{output_path}/narrative_{narrative}.txt", "w") as f:
            narrative_df = filter_data_by_narrative_id(df, narrative)
            for i, row in narrative_df.iterrows():
                f.write(f"User ID: {row['user_id']}\n")
                f.write(f"Summarization: {row['value']['summarization']}\n\n")


def visualize_conv_user_elicitation_based_on_narrative_and_invester(df, output_dir):
    df.sort_values(by=["user_id_uuid", "timestamp"], inplace=True)
    narratives = ["0", "1", "2"]
    for narrative in narratives:
        narrative_df = filter_data_by_narrative_id(df, narrative)
        narrative_df = (
            narrative_df.groupby("user_id_uuid", group_keys=False)
            .apply(concat_conv_based_on_user_id)
            .reset_index()
        )
        narrative_df.columns = ["user_id_uuid", "conversation"]
        narrative_df["conversation"] = narrative_df["conversation"].apply(
            lambda x: '"' + x.replace('"', '""') + '"'
        )
        narrative_df.to_excel(
            f"{output_dir}/narrative_{narrative}.xlsx", index=False, header=True
        )
        # with open(f"{output_dir}/narrative_{narrative}.txt", "w") as f:
        #     narrative_df = filter_data_by_narrative_id(df, narrative)
        #     narrative_df = (
        #         narrative_df.groupby("user_id_uuid", group_keys=False).apply(concat_conv_based_on_user_id).reset_index()
        #     )
        #     narrative_df.columns = ["user_id_uuid", "conversation"]
        #     for i, row in narrative_df.iterrows():
        #         f.write(f"User ID: {row['user_id_uuid']}\n")
        #         f.write(f"Conversation: {row['conversation']}\n\n")


def visualize_conv_financial_decision_based_on_narrative_and_invester(df, output_dir):
    df.sort_values(by=["timestamp"], inplace=True)
    output_path = os.path.join(output_dir, "financial_decision")
    narratives = ["0", "1", "2"]
    for narrative in narratives:
        with open(f"{output_path}/narrative_{narrative}.txt", "w") as f:
            narrative_df = filter_data_by_narrative_id(df, narrative)
            narrative_df = (
                narrative_df.groupby(["user_id", "company_name"], group_keys=False)
                .apply(concat_conv_based_on_user_id)
                .reset_index()
            )
            # Update the columns to include 'company_name'
            narrative_df.columns = ["user_id", "company_name", "conversation"]
            for i, row in narrative_df.iterrows():
                f.write(f"User ID: {row['user_id']}\n")
                # Add the company type to the output
                f.write(f"Company Type: {row['company_name']}\n")
                f.write(f"Conversation: {row['conversation']}\n\n")


def expand_dict_column(df, column, df_name=None):
    expanded_values = pd.json_normalize(df[column])
    if df_name == "financial_decision":
        expanded_values.columns = ["company_name", "role", "content"]
    return pd.concat([df, expanded_values], axis=1)


def get_context_list(synthetic_data_path):
    # Load data from the synthetic data file
    with open(synthetic_data_path, "r") as f:
        data = [json.loads(line) for line in f]

    return data


def get_company_label(context_info_list):
    company_label_dict = {}
    for i in range(len(context_info_list)):
        for stock_data in context_info_list[i]["data"]:
            company_label_dict[stock_data["short_name"]] = stock_data["label"]
    return company_label_dict


def binary_accuracy_round(round_evaluation_eval):
    acc_df = round_evaluation_eval.copy()
    acc_df["binary_label"] = acc_df["label"].apply(lambda x: 1 if x > 2 else 0)
    acc_df["binary_lilelihood"] = acc_df["likelihood"].apply(
        lambda x: 1 if x > 4 else (0 if x < 4 else None)
    )
    acc_df.dropna(inplace=True)
    return accuracy_score(acc_df["binary_label"], acc_df["binary_lilelihood"])


def ranking_accuracy(final_ranking, method="kendall"):
    assert method in [
        "kendall",
        "spearman",
        "tauap",
    ], "Method should be either kendall or spearman"
    acc_df = final_ranking.copy()
    if method == "kendall":
        acc_df["kendall"] = acc_df.apply(
            lambda x: kendalltau(x["ranking"], x["gold_ranking"]).statistic, axis=1
        )
        return acc_df["kendall"].mean()
    elif method == "tauap":
        pass
    else:
        acc_df["spearman"] = acc_df.apply(
            lambda x: spearmanr(x["ranking"], x["gold_ranking"]).statistic, axis=1
        )
        return acc_df["spearman"].mean()
