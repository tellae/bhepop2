import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from os import path

default_plot_title_format = "Modality {modality} from attribute {attribute}"


def analyse_enriched_populations(populations, distributions, observed_name, modalities, output_folder, feature_column_name="feature", plots=True, error_table=True, plots_title_format=None):
    """
    Generate several analysis comparing given population(s) to the original distributions.

    Analysis is realised on the given modalities, which must be a subset of the modalities used for enrichment
    (and thus available in the population(s) and distributions).

    The following analysis are provided:
        - Graphs comparing the deciles of the population(s) to those of the distributions (one per modality)
        - A table describing the error of the population(s) in comparison to the distributions (one line per modality),
        order by number of individuals in the modality

    :param populations: dict containing populations, with population name as keys
    :param distributions: reference distributions
    :param observed_name: name of the distributions data (source)
    :param modalities: analysed modalities
    :param output_folder: folder where outputs are generated
    :param feature_column_name: name of the column containing the added feature
    :param plots: boolean indicating if plots should be generated
    :param error_table: boolean indicating if error table should be generated
    :param plots_title_format: format of the graph titles, can use 'observed_name', 'attribute' and 'modality' variables

    :return: analysis DataFrame
    """

    # filter distributions with given modalities
    all_modalities = [modality for attribute_modalities in modalities.values() for modality in attribute_modalities] + ["all"]
    distributions = distributions[distributions["modality"].isin(all_modalities)]

    # get populations names
    populations_names = list(populations.keys())

    # compute analysis table
    analysis_df = get_analysis_table(populations, distributions, observed_name, modalities, feature_column_name)

    # generate analysis objects
    if plots:
        generate_analysis_plots(analysis_df, populations_names, observed_name, modalities, output_folder, title_format=plots_title_format)
    if error_table:
        generate_analysis_error_table(analysis_df, populations_names, observed_name, modalities, populations["base"], output_folder)

    return analysis_df


# error table generation

def generate_analysis_error_table(analysis_df, populations_names, observed_name, modalities, pop, output_folder, export_csv=True):
    """
    Generate a table describing how analysed populations deviate from the original distributions.

    :param analysis_df: analysis DataFrame (see get_analysis_table function)
    :param populations_names: name of the analysed populations
    :param observed_name: name of the distributions data (source)
    :param modalities: analysed modalities
    :param pop:
    :param output_folder: folder where outputs are generated
    :param export_csv: boolean a csv export should be realised

    :return: error table DataFrame
    """

    # evaluate distance to distributions
    for source in populations_names:
        analysis_df[f"{source}_perc_error"] = abs(
            (analysis_df[source] - analysis_df[observed_name]) / analysis_df[observed_name]
        )
    error_analysis_df = analysis_df.groupby(["attribute", "modality"], as_index=False).agg(
        {f"{source}_perc_error": "mean" for source in populations_names}
    )

    # count number of occurrence of each modality and add it as a "number" column, sort descending
    count = [pop.groupby([attribute])[attribute].count().rename("number") for attribute in modalities.keys()]
    count = pd.concat(count)
    count["all"] = len(pop)
    error_analysis_df = error_analysis_df.merge(count, how="left", left_on="modality", right_index=True)
    error_analysis_df.sort_values(["number"], ascending=False, inplace=True)

    # export to csv if asked
    if export_csv:
        error_analysis_df.to_csv(f"{output_folder}/analysis_error.csv", sep=";", index=False)

    return error_analysis_df


# analysis plots generation

def generate_analysis_plots(analysis_df, populations_names, observed_name, modalities, output_folder, title_format=None):
    """
    Graphs comparing the deciles of the population(s) to those of the distributions (one per modality).

    :param analysis_df: analysis DataFrame (see get_analysis_table function)
    :param modalities: analysed modalities
    :param populations_names: name of the analysed populations
    :param observed_name: name of the distributions data (source)
    :param output_folder: folder where outputs are generated
    :param title_format: format of the graph titles, can use 'observed_name', 'attribute' and 'modality' variables
    """

    # plot analysis for global distributions
    plot_analysis_compare(analysis_df, "all", "all", observed_name=observed_name,
                          populations_names=populations_names, title_format=title_format).write_image(
        path.join(output_folder, "all-all.png")
    )

    # plot analysis by attribute
    for attribute in modalities.keys():
        for modality in modalities[attribute]:
            plot_analysis_compare(
                analysis_df, attribute, modality, observed_name=observed_name, populations_names=populations_names, title_format=title_format
            ).write_image(path.join(output_folder, f"{attribute}-{modality}.png"))


def plot_analysis_compare(
    df: pd.DataFrame,
    attribute: str,
    modality: str,
    observed_name: str,
    populations_names: list,
    title_format: str = None
):
    """
    Comparison plot between reference data and simulation

    :param df: analysis DataFrame
    :param attribute:
    :param modality:
    :param observed_name:
    :param populations_names: list of population names
    :param title_format: format of the graph titles, can use 'observed_name', 'attribute' and 'modality' variables

    :return: Plotly figure
    """

    if title_format is None:
        title_format = default_plot_title_format
    title = title_format.format(**{
        "observed_name": observed_name,
        "attribute": attribute,
        "modality": modality
    })

    df = df.copy()[(df["attribute"] == attribute) & (df["modality"] == modality)]
    fig = px.line(x=df[observed_name], y=df[observed_name], color_discrete_sequence=["black"])
    # add simulated names
    for simulated in populations_names:
        fig.add_trace(
            go.Scatter(x=df[observed_name], y=df[simulated], mode="markers", name=simulated)
        )
    # configure plot
    fig.update_layout(
        title=title,
        xaxis_title=f"Observation ({observed_name})",
        yaxis_title="Simulation",
    )

    return fig


# processing data (populations, distributions) for analysis

def get_analysis_table(populations, distributions, observed_name, modalities, feature_column_name="feature"):
    f"""
    Create a table comparing population deciles to distribution deciles, per modality.

    The resulting DataFrame contains the following columns:
        - attribute: attribute name
        - modality: modality name
        - decile: D1, D2, .. , D9
        - one column with the observed_name value
        + one column for each population name

    :param populations: dict containing populations, with population name as keys 
    :param distributions: reference distributions
    :param observed_name: name of the distributions data (source)
    :param modalities: analysed modalities
    :param feature_column_name: name of the column containing the added feature
    
    :return: analysis DataFrame
    """
    if len(populations) == 0:
        raise ValueError("No population to analyse")
    populations_names = list(populations.keys())

    # format distributions for analysis
    distributions_df = format_distributions_for_analysis(distributions)
    distributions_df["source"] = observed_name

    # add population analysis dataframes
    analysis_list = [distributions_df]
    for name, population in populations.items():
        population_df = compute_distributions_by_attribute(population, modalities, feature_column_name)
        population_df["source"] = name
        analysis_list.append(population_df)

    # create an analysis dataframe containing all deciles
    analysis_df = pd.concat(analysis_list)

    # pivot data to get final dataframe
    analysis_df = analysis_df.pivot(
        columns="source", index=["attribute", "modality", "decile"]
    ).reset_index()
    analysis_df.columns = ["attribute", "modality", "decile", observed_name] + populations_names

    return analysis_df


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


def compute_distributions_by_attribute(pop: pd.DataFrame, modalities,
                                       feature_column_name: str = "feature") -> pd.DataFrame:
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


def compute_distribution(df: pd.DataFrame, feature_column_name: str = "feature") -> pd.DataFrame:
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
