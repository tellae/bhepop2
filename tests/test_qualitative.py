from bhepop2.max_entropy_enrich_qualitative import QualitativeEnrichment

import numpy as np
import pyarrow.feather as feather
import pytest
import pandas as pd
from bhepop2.functions import *


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


@pytest.fixture(scope="session")
def pop_synt_men_nantes():
    synt_pop = feather.read_feather("data/inputs/pop_synt_men_nantes.feather")
    synt_pop = synt_pop.drop(['VOIT'], axis=1)
    synt_pop = synt_pop.dropna()
    return synt_pop


@pytest.fixture(scope="session")
def distributions(pop_synt_men_nantes):
    attributes = list(pop_synt_men_nantes.columns[:-1])
    marginal_distribution = pd.concat(list(map(lambda a: build_cross_table(pop_synt_men_nantes, [a, 'Voit_rec']), attributes)))
    marginal_distribution = marginal_distribution.loc[~marginal_distribution.index.duplicated(keep='first')]
    marginal_distribution.loc["all", "attribute"] = "all"
    return marginal_distribution


def test_qualitative_enrich(
    pop_synt_men_nantes,
    distributions,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes,
):
    modalities = infer_modalities_from_distributions(distributions)
    modalities["Voit_rec"] = ["0voit", "1voit", "2voit", "3voit"]
    test = compute_crossed_modalities_frequencies(pop_synt_men_nantes, modalities)


    def calc(x):
        return x["probability"] / distributions.loc["all", x["Voit_rec"]]

    test["prob_cond"] = test.apply(calc, axis=1)

    test.apply(
        calc,
        axis=1,
    )

    # le test commence ici

    synt_pop_defected = pop_synt_men_nantes.drop(['Voit_rec'], axis=1)

    enrich_class = QualitativeEnrichment(
        synt_pop_defected, distributions, seed=test_seed
    )

    # Run optimisation
    enrich_class.optimise()


    # Assign data to synthetic population
    pop = enrich_class.assign_feature_value_to_pop()

    pop.head()
