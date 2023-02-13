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


def validate_distributions(distributions):
    """
    Validate the format and contents of the given distribution.

    :param distributions: distribution DataFrame
    :raises: AssertionError
    """

    # we could validate the distributions (positive, monotony ?)
    assert {*["D{}".format(i) for i in range(1, 10)], "attribute", "modality"} <= set(
        distributions.columns
    )


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
    probs_df["prob"] = probs_df.apply(
        lambda x: interpolate_feature_prob(x["feature"], distribution),
        axis=1,
    )

    return probs_df


def interpolate_feature_prob(feature_value: float, distribution: list):
    """
    Linear interpolation of a feature value probability.

    :param feature_value: value of feature to interpolate
    :param distribution: feature values for each decile from 1 to 10 (without value for 0)

    :return: probability of being lower than the input feature value
    """
    distribution = [0] + distribution
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


def validate_population(population: pd.DataFrame, modalities):
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
        assert population[attribute].isin(modalities[attribute]).all()


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


# check functions


def compute_rq(model, nb_modalities, K):
    I = np.identity(nb_modalities)
    A = model.F.A
    A_eq = np.concatenate([A, I], axis=1)

    c = np.concatenate([np.zeros(np.shape(A)[1]), np.ones(nb_modalities)], axis=None)

    res = linprog(c, A_eq=A_eq, b_eq=K, method="simplex")

    return res
