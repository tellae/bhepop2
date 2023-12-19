from bhepop2.functions import infer_modalities_from_distributions
from bhepop2.analysis import QualitativeAnalysis, QuantitativeAnalysis

import pandas as pd


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
    table_percentage_attribute2 = (
        table_numbers.transpose().sum() / table_numbers.transpose().sum().sum()
    )
    table_percentage = table_numbers / table_numbers.sum()
    table_percentage["all"] = table_percentage_attribute2
    table_percentage = table_percentage.transpose()
    table_percentage["modality"] = table_percentage.index
    table_percentage["attribute"] = name_attribute1

    return table_percentage


def test_quantitative_analysis(
    filosofi_distributions_nantes, expected_enriched_population_nantes, test_modalities, tmp_dir
):
    analysis = QuantitativeAnalysis(
        {"qualitative": expected_enriched_population_nantes},
        test_modalities,
        "feature",
        filosofi_distributions_nantes,
        output_folder=tmp_dir,
    )

    analysis.generate_analysis_plots()
    analysis.generate_analysis_error_table()


def test_qualitative_analysis(pop_synt_men_nantes, tmp_dir):
    # synth pop
    pop = pop_synt_men_nantes.copy()

    # distributions
    attributes = list(pop.columns[:-1])
    marginal_distribution = pd.concat(
        list(map(lambda a: build_cross_table(pop, [a, "Voit_rec"]), attributes))
    )
    # removing multiple 'all'
    marginal_distribution = marginal_distribution.loc[
        ~marginal_distribution.index.duplicated(keep="first")
    ]
    marginal_distribution.loc["all", "attribute"] = "all"
    marginal_distribution = marginal_distribution.reset_index(drop=True)

    modalities = infer_modalities_from_distributions(marginal_distribution)

    analysis = QualitativeAnalysis(
        {"qualitative": pop}, modalities, "Voit_rec", marginal_distribution, output_folder=tmp_dir
    )

    analysis.generate_analysis_plots()
