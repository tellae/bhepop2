import pandas as pd
import numpy as np
from scipy.optimize import linprog


# generic functions


def get_attributes(modalities: dict) -> list:
    """
    Get attributes list from dictionary of modalities

    :param modalities:

    :return: attributes
    """
    return list(modalities.keys())


def modality_feature(attribute, modality) -> callable:
    """
    Create a function that checks if a sample belongs to the given attribute and modality.

    :param attribute: attribute value
    :param modality: modality value

    :return: feature checking function
    """

    def feature(x):
        return x[attribute] == modality

    return feature


# distribution functions


def validate_distributions(distributions: pd.DataFrame, attribute_selection, mode):
    """
    Validate the format and contents of the given distribution.

    :param distributions: distribution DataFrame
    :param attribute_selection: list of attributes to keep in the distribution, or None
    :param mode: "qualitative" or "quantitative"
    :raises: AssertionError
    """

    assert not distributions.empty, "Empty distributions table provided"

    if mode == "quantitative":
        # we could validate the distributions columns (positive, monotony ?)
        assert {*["D{}".format(i) for i in range(1, 10)], "attribute", "modality"} <= set(
            distributions.columns
        ), "Distributions table lacks the required columns"
    elif mode == "qualitative":
        assert "attribute" in distributions.columns and "modality" in distributions.columns
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    if attribute_selection is not None:
        # check that the distributions contain the selected attributes
        assert set(attribute_selection + ["all"]) <= set(
            distributions["attribute"]
        ), "Distributions table does not include selected attributes"


def filter_distributions_and_infer_modalities(distributions: pd.DataFrame, attribute_selection):
    """
    Filter distributions table with attribute selection and infer modalities.

    :param distributions: distribution DataFrame
    :param attribute_selection: list of attributes to keep in the distribution, or None

    :return: filtered distribution Dataframe, { attribute: [modalities] } dict
    """
    # make a copy of the distributions
    distributions = distributions.copy()

    if attribute_selection is not None:
        # filter distributions attributes
        distributions = distributions[
            distributions["attribute"].isin(attribute_selection + ["all"])
        ]

    # infer attributes and their modalities from the filtered distribution
    modalities = infer_modalities_from_distributions(distributions)

    return distributions, modalities


def infer_modalities_from_distributions(distributions: pd.DataFrame):
    """
    Infer attributes and their modalities from the given distributions.

    :param distributions: distributions DataFrame

    :return: dict of attributes and their modalities, { attribute: [modalities] }
    """

    # group by attribute and get all modality values
    modalities = distributions.groupby("attribute")["modality"].apply(list).to_dict()

    # remove global distribution
    if "all" in modalities:
        del modalities["all"]

    return modalities


# TODO : rename with reference to quantitative nature of distribution. Or move to quantitative class
def compute_feature_values(
    distribution: pd.DataFrame, relative_maximum: float, delta_min=None
) -> list:
    """
    Compute the list of feature values that will define the assignment intervals.

    The distributions do not give the knowledge of the minimum and maximum
    feature values, so we have to choose them.
    The minimum is the same for all distributions, it is directly equal to the abs_first_value parameter.
    The maximum is computed by multiplying the relative_maximum parameter to the last value of each distribution.

    :param distribution: dataframe of distribution
    :param relative_maximum: multiplicand applied to compute the last feature value of each distribution
    :param delta_min: minimum delta between two feature values. None to keep all values.

    :return: list of feature values
    """
    deciles = distribution.copy()

    # set maximum values
    deciles["D10"] = deciles["D9"] * relative_maximum

    # restraint columns to distribution values and get vector
    deciles = deciles[list(map(lambda a: "D" + str(a), list(range(1, 11))))]
    vec_all = list(deciles.to_numpy().flatten())

    # sort values vector
    vec_all.sort()

    # remove close values using delta min
    if delta_min is not None:
        last_value = vec_all[0]
        filtered_vec = [last_value]
        for val in vec_all[1:]:
            if val - last_value >= delta_min:
                filtered_vec.append(val)
                last_value = val
        vec_all = filtered_vec

    return vec_all


def get_feature_from_qualitative_distribution(distribution: pd.DataFrame):
    """
    Get feature values from the given distributions.

    :param distribution: distribution DataFrame

    :return: list of possible values for the qualitative feature
    """

    features = list(distribution.columns)
    features.remove("attribute")
    features.remove("modality")

    assert (distribution[features].apply(lambda row: np.isclose(row.sum(), 1), axis=1)).all()

    return features


def compute_features_prob(feature_values: list, distribution: list):
    """
    Create a DataFrame containing probabilities for the given feature values.

    :param feature_values: list of feature values
    :param distribution: list of distribution values

    :return: DataFrame of feature probabilities
    """
    # set features column
    probs_df = pd.DataFrame({"feature": feature_values})

    # compute prob of being in each feature interval
    probs_df["cumulative"] = probs_df.apply(
        lambda x: interpolate_feature_prob(x["feature"], distribution),
        axis=1,
    )

    probs_df["tmp"] = probs_df["cumulative"].shift(1)
    probs_df = probs_df.fillna(0)
    probs_df["prob"] = probs_df["cumulative"] - probs_df["tmp"]
    probs_df.drop(columns=["cumulative", "tmp"], inplace=True)

    return probs_df


def interpolate_feature_prob(feature_value: float, distribution: list):
    """
    Linear interpolation of a feature value probability.

    First and last distribution values represent minimum and maximum values
    that can be taken.

    :param feature_value: value of feature to interpolate
    :param distribution: feature values for each decile from 0 to 10

    :return: probability of being lower than the input feature value
    """

    if feature_value > distribution[10]:
        return 1
    if feature_value < distribution[0]:
        return 0
    decile_top = 0
    while feature_value > distribution[decile_top]:
        decile_top += 1

    interpolation = (feature_value - distribution[decile_top - 1]) * (
        decile_top * 0.1 - (decile_top - 1) * 0.1
    ) / (distribution[decile_top] - distribution[decile_top - 1]) + (decile_top - 1) * 0.1

    return interpolation


# population functions


def validate_population(population: pd.DataFrame, modalities: dict):
    """
    Validate the format and contents of the given population.

    Check that the population is compatible with the chosen modalities.

    :param population: distribution DataFrame
    :param modalities:
    :raises: AssertionError
    """

    attributes = get_attributes(modalities)

    assert {*attributes} <= set(population.columns)

    for attribute in attributes:
        assert population[attribute].isin(modalities[attribute]).all(), (
            f"Population validation: one of the modality values was not "
            f"found in distributions for the attribute '{attribute}'"
        )


def compute_crossed_modalities_frequencies(
    population: pd.DataFrame, modalities: dict
) -> pd.DataFrame:
    """
    Compute the frequency of each crossed modality present in the population.

    Columns other than attributes are removed from the result DataFrame, and a
    'probability' column is added.

    :param population: population DataFrame
    :param modalities: modalities dict

    :return: DataFrame of crossed modalities frequencies
    """

    attributes = get_attributes(modalities)

    # group by attributes and count the number of individuals, then divide by total
    population["count"] = 1
    freq_df = population.groupby(attributes, as_index=False)["count"].sum()
    freq_df["probability"] = freq_df["count"] / sum(freq_df["count"])
    freq_df = freq_df[attributes + ["probability"]]

    # remove count column
    population.drop("count", axis=1, inplace=True)

    return freq_df


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


# check functions


def compute_rq(model, nb_modalities, K):
    I = np.identity(nb_modalities)
    A = model.F.A
    A_eq = np.concatenate([A, I], axis=1)

    c = np.concatenate([np.zeros(np.shape(A)[1]), np.ones(nb_modalities)], axis=None)

    res = linprog(c, A_eq=A_eq, b_eq=K, method="simplex")

    return res
