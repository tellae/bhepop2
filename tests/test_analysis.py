import pandas as pd

from bhepop2.analysis import *
import pyarrow.feather as feather
from bhepop2.functions import infer_modalities_from_distributions, get_feature_from_qualitative_distribution

def test_analyse_enriched_population(
    filosofi_distributions_nantes, expected_enriched_population_nantes, test_modalities, tmp_dir
):
    populations = {"enriched": expected_enriched_population_nantes}

    analyse_enriched_populations(
        populations,
        filosofi_distributions_nantes,
        "Filosofi",
        test_modalities,
        tmp_dir,
        plots=False,
    )


def test_compute_distribution(expected_enriched_population_nantes):
    """
    Test that computed distribution is exactly the same as expected
    """
    expected = [
        {"feature": 9890.6, "decile": "D1"},
        {"feature": 13190.0, "decile": "D2"},
        {"feature": 15779.0, "decile": "D3"},
        {"feature": 18314.0, "decile": "D4"},
        {"feature": 20865.0, "decile": "D5"},
        {"feature": 23738.0, "decile": "D6"},
        {"feature": 27290.0, "decile": "D7"},
        {"feature": 32093.8, "decile": "D8"},
        {"feature": 41201.4, "decile": "D9"},
    ]

    assert (
        compute_distribution(expected_enriched_population_nantes).round(1).to_dict(orient="records")
        == expected
    )


def build_cross_table(pop: pd.DataFrame, names_attribute: list):
    """


    Parameters
    ----------
    pop : DataFrame synthesis population
    names_attribute: list of two strings
           name of attribute1 and name of attribute 2

    Returns
    -------
    table_percentage : DataFrame
          proportion of modalities of attribute 2 given attribute 1


    """

    name_attribute1 = names_attribute[0]
    name_attribute2 = names_attribute[1]
    table_numbers = pd.crosstab(pop[name_attribute2], pop[name_attribute1])
    table_percentage_attribute2 = table_numbers.transpose().sum() / table_numbers.transpose().sum().sum()
    table_percentage = table_numbers / table_numbers.sum()
    table_percentage['all'] = table_percentage_attribute2
    table_percentage = table_percentage.transpose()
    table_percentage['modality'] = table_percentage.index
    table_percentage['attribute'] = name_attribute1

    return table_percentage


#      attribute    modality decile       feature    source
# 0          all         all     D1  10303.478261  Filosofi
# 1         size      1_pers     D1   9794.000000  Filosofi
# 2         size      2_pers     D1  12176.000000  Filosofi
# 3         size      3_pers     D1  10583.500000  Filosofi
# 4         size      4_pers     D1  10740.476190  Filosofi
# ..         ...         ...    ...           ...       ...
# 175        age       50_59     D9  46658.000000  Filosofi
# 176        age       60_74     D9  48548.000000  Filosofi
# 177        age  75_or_more     D9  43945.000000  Filosofi
# 178  ownership       Owner     D9  50060.000000  Filosofi
# 179  ownership      Tenant     D9  29860.476190  Filosofi


def get_analysis_table( populations, distributions, observed_name, modalities, feature_column_name="feature"):
    if len(populations) == 0:
        raise ValueError("No population to analyse")
    populations_names = list(populations.keys())

    # format distributions for analysis
    distributions_df = format_distributions_for_analysis(distributions)
    distributions_df["source"] = observed_name

    # add population analysis dataframes
    analysis_list = [distributions_df]
    for name, population in populations.items():
        population_df = compute_distributions_by_attribute(
            population, modalities, feature_column_name
        )
        population_df["source"] = name
        analysis_list.append(population_df)

    # create an analysis dataframe containing all deciles
    analysis_df = pd.concat(analysis_list)

    # pivot data to get final dataframe
    analysis_df = analysis_df.reset_index(drop=True)
    print(analysis_df)
    analysis_df = analysis_df.pivot(
        columns="source", index=["attribute", "modality", "feature"]
    ).reset_index()
    columns = list(analysis_df.columns.get_level_values(1))
    columns[0], columns[1], columns[2] = "attribute", "modality", "feature"
    analysis_df.columns = columns

    return analysis_df

def compute_distributions_by_attribute(
    pop: pd.DataFrame, modalities, feature_column_name: str = "feature"
) -> pd.DataFrame:

    df_analysis = compute_distribution(pop, feature_column_name)
    df_analysis["attribute"] = "all"
    df_analysis["modality"] = "all"
    # distribution of each modality
    for attribute in modalities.keys():
        for modality in modalities[attribute]:
            distribution = compute_distribution(
                pop[pop[attribute] == modality], feature_column_name
            )
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

    df["key"] = 1

    res = df.groupby(feature_column_name)["key"].agg("count")

    df = pd.DataFrame({
        "feature": res.index,
        "proportion": res.values
    })
    df["proportion"] = df["proportion"] / df["proportion"].sum()

    return df


def format_distributions_for_analysis(distributions):
    feature_values = get_feature_from_qualitative_distribution(distributions)
    distributions = distributions[
        ["attribute", "modality"] + feature_values
    ]

    distributions_formated = distributions.melt(
        id_vars=["attribute", "modality"],
        value_vars=feature_values,
        value_name="proportion",
        var_name="feature",
    )
    distributions_formated["source"] = "original"

    return distributions_formated

def plot_analysis_compare(
    df: pd.DataFrame,
    attribute: str,
    modality: str,
    observed_name: str,
    populations_names: list,
    title_format: str = None,
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
    title = title_format.format(
        **{"observed_name": observed_name, "attribute": attribute, "modality": modality}
    )

    df = df.copy()[(df["attribute"] == attribute) & (df["modality"] == modality)]
    print(df)

    # df = px.data.tips()
    print(df)
    fig = px.histogram(df, x="feature", y=["distributions", populations_names[0]],
                       barmode='group',
                       histnorm="percent")

    # configure plot
    fig.update_layout(
        title=title,
        xaxis_title="Values",
        yaxis_title="Distribution (%)",
    )

    return fig


def test_qualitative_analysis():
    # synth pop
    pop = feather.read_feather("data/inputs/pop_synt_men_nantes.feather")
    pop = pop.drop(['VOIT'], axis=1)  # this attribute is redundant with the attribute 'Voit_rec'
    pop = pop.dropna()  # POV  some values are set to None : it was detected thanks to the functions.validatee_population

    # distributions
    attributes = list(pop.columns[:-1])
    marginal_distribution = pd.concat(list(map(lambda a: build_cross_table(pop, [a, 'Voit_rec']), attributes)))
    # removing multiple 'all'
    marginal_distribution = marginal_distribution.loc[~marginal_distribution.index.duplicated(keep='first')]
    marginal_distribution.loc["all", "attribute"] = "all"
    marginal_distribution = marginal_distribution.reset_index(drop=True)

    modalities = infer_modalities_from_distributions(marginal_distribution)

    table = get_analysis_table({ "qualitative": pop }, marginal_distribution, "distributions", modalities=modalities, feature_column_name="Voit_rec")
    print(table)

    plot_analysis_compare(table, "AGEREVQb_rec", "cat1", "distributions", ["qualitative"]).write_image(path.join("plots/", "test.png"))





