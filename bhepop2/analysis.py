import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from os import path


def format_distributions_for_analysis(distributions):
    distributions = distributions[["attribute", "modality", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]]
    distributions_formated = distributions.melt(
        id_vars=["attribute", "modality"],
        value_vars=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
        value_name="feature",
        var_name="decile",
    )
    distributions_formated["source"] = "original"
    return distributions_formated


def compute_distributions_by_attribute(pop: pd.DataFrame, modalities, feature_column_name: str = "feature") -> pd.DataFrame:
    df_analysis = compute_distribution(pop, feature_column_name)
    df_analysis["attribute"] = "all"
    df_analysis["modality"] = "all"

    # distribution of each modality
    for attribute in modalities.keys():
        for modality in modalities[attribute]:
            distribution = compute_distribution(pop[pop[attribute] == modality], feature_column_name)
            distribution["attribute"] = attribute
            distribution["modality"] = modality

            df_analysis = pd.concat([df_analysis, distribution])
    df_analysis["source"] = "enriched"
    return df_analysis


def analyse_enriched_populations(populations, distributions, modalities, output_folder, observation_name, feature_column_name="feature"):

    if len(populations) == 0:
        raise ValueError("No population to analyse")
    simulated_names = list(populations.keys())

    # format distributions for analysis
    distributions_df = format_distributions_for_analysis(distributions)
    distributions_df["source"] = observation_name

    # add population analysis dataframes
    analysis_list = [distributions_df]
    for name, population in populations.items():
        population_df = compute_distributions_by_attribute(population, modalities, feature_column_name)
        population_df["source"] = name
        analysis_list.append(population_df)

    # create an analysis dataframe containing all deciles
    analysis_df = pd.concat(analysis_list)

    # pivot data to get
    analysis_df = analysis_df.pivot(
        columns="source", index=["attribute", "modality", "decile"]
    ).reset_index()
    analysis_df.columns = ["attribute", "modality", "decile", observation_name] + simulated_names

    # plot analysis for global distributions
    plot_analysis_compare(analysis_df, "all", "all", observed_name=observation_name, simulated_name=simulated_names).write_image(
        path.join(output_folder, "analysis_all_all.png")
    )

    # plot analysis by attribute
    for attribute in modalities.keys():
        for modality in modalities[attribute]:
            plot_analysis_compare(
                analysis_df, attribute, modality, observed_name=observation_name, simulated_name=simulated_names
            ).write_image(path.join(output_folder, f"analysis_{attribute}_{modality}.png"))


def compute_distribution(df: pd.DataFrame, feature_column_name:str = "feature") -> pd.DataFrame:
    """
    Compute decile distribution on one the DataFrame's columns.

    :param df: analysed DataFrame
    :param feature_column_name: name of the column to compute distributions on

    :return: dataframe of deciles
    """
    return pd.DataFrame(
        {
            "feature": np.percentile(
                df[feature_column_name],
                np.arange(0, 100, 10),
            )[1:],
            "decile": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
        }
    )


def plot_analysis_compare(
    df: pd.DataFrame,
    attribute: str,
    modality: str,
    observed_name: str,
    simulated_name: list,
):
    """
    Comparison plot between reference data and simulation

    :param df: analysis DataFrame
    :param attribute:
    :param modality:
    :param observed_name:
    :param simulated_name:

    :return: Plotly figure
    """
    df = df.copy()[(df["attribute"] == attribute) & (df["modality"] == modality)]
    fig = px.line(x=df[observed_name], y=df[observed_name], color_discrete_sequence=["black"])
    # add simulated names
    for simulated in simulated_name:
        fig.add_trace(
            go.Scatter(x=df[observed_name], y=df[simulated], mode="markers", name=simulated)
        )
    # configure plot
    fig.update_layout(
        title=f"Modality {modality} from attribute {attribute}",
        xaxis_title=f"Observation ({observed_name})",
        yaxis_title="Simulation",
    )

    return fig
