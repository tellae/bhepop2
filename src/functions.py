from itertools import product
from scipy.optimize import linprog
import maxentropy
import pandas as pd
import numpy as np
import math
import utils

DECILE_0 = 0
DECILE_10 = 1.5


def compute_p_r(vec_all, df_imputed: pd.DataFrame, code_insee: str) -> list:
    """
    Compute p_R

    :param vec_all:
    :param df_imputed:

    return: p_R
    """
    distribution_all = df_imputed.query(f"commune_id == '{code_insee}'")

    total_population_decile = [
        distribution_all["q1"].iloc[0],
        distribution_all["q2"].iloc[0],
        distribution_all["q3"].iloc[0],
        distribution_all["q4"].iloc[0],
        distribution_all["q5"].iloc[0],
        distribution_all["q6"].iloc[0],
        distribution_all["q7"].iloc[0],
        distribution_all["q8"].iloc[0],
        distribution_all["q9"].iloc[0],
        max(vec_all),  # maximum value of all deciles
    ]
    # linear extrapolation of these 190 incomes from the total population deciles
    p_R = pd.DataFrame({"income": vec_all})
    p_R["proba1"] = p_R.apply(
        lambda x: utils.interpolate_income(x["income"], total_population_decile), axis=1
    )

    return p_R


def compute_vec_all(distribution, decile_0: float = DECILE_0, decile_10: float = DECILE_10) -> list:
    """
    Compute vector of values in distribution

    :param distribution: dataframe of distribution
    :param decile_0: value of decile 0
    :param decile_10: value of decile 10 relative to decile 9

    return: list of values
    """
    decile_total = distribution.copy()
    decile_total["D0"] = decile_0
    decile_total["D10"] = decile_total["D9"] * decile_10
    decile_total = decile_total[
        ["modality"] + list(map(lambda a: "D" + str(a), list(range(0, 11))))
    ]

    # get all deciles and sort values
    vec_all = []
    for index, row in decile_total.iterrows():
        for r in list(map(lambda a: "D" + str(a), list(range(1, 11)))):
            vec_all.append(row[r])
    vec_all.sort()

    return vec_all


def compute_distribution(
    df_attributes: pd.DataFrame, df_imputed: pd.DataFrame, code_insee: str, modalities: dict
) -> pd.DataFrame:
    """
    Format income and imputed data if needed

    :param df_attributes: income distribution per attributes
    :param df_imputed: income distribution imputed for all
    :param code_insee: insee code of selected city
    :param modalities: dictionary of modalities

    return: dataframe of formated distribution
    """
    attributes = get_attributes(modalities)

    filosofi = df_attributes.query(f"commune_id == '{code_insee}'")
    if len(filosofi["attribute"].unique()) < len(attributes):
        print("ERROR : need imputation of income")
        # TODO implement an imputation method when data is missing

    filosofi.rename(
        columns={
            "q1": "D1",
            "q2": "D2",
            "q3": "D3",
            "q4": "D4",
            "q5": "D5",
            "q6": "D6",
            "q7": "D7",
            "q8": "D8",
            "q9": "D9",
        },
        inplace=True,
    )
    return filosofi


# run assignment


def run_assignment(external_date, vec_all_incomes, grouped_pop, modalities):

    # create dictionary of constraints (element of eta_total in R code)
    constraint = create_constraints(modalities, external_date, vec_all_incomes, grouped_pop)

    # optimisation (maxentropy)

    samplespace_reducted, f, function_prior_prob = create_samplespace_and_features(
        modalities, grouped_pop
    )

    # build K

    model_with_apriori = create_model(f, samplespace_reducted, function_prior_prob)
    incomes = [0, 1, 2]
    res = pd.DataFrame()
    # loop on incomes
    for i in incomes:
        print("Running model for income " + str(i))
        # we do build the model again because it seemed to break after a failed fit

        run_model_on_income(model_with_apriori, i, modalities, constraint)
        res.loc[:, i] = model_with_apriori.probdist()

        # need to reset dual for next iterations !
        model_with_apriori.resetparams()

    return res


def get_attributes(modalities: dict) -> list:
    """
    Get attributes list from dictionary of modalities

    :param modalities:

    return: attributes
    """
    return list(modalities.keys())


# prepare data
def compute_crossed_probabilities(synthetic_pop: pd.DataFrame, modalities: dict) -> pd.DataFrame:
    """
    Computed crossed probabilities on attribues

    :param synthetic_pop: dataframe of population
    :param modalities: dictionary of modalities

    return: dataframe of crossed probabilities
    """
    attributes = get_attributes(modalities)

    synthetic_pop["count"] = 1
    group = synthetic_pop.groupby(attributes, as_index=False)["count"].sum()
    group["probability"] = group["count"] / sum(group["count"])
    group = group[attributes + ["probability"]]

    return group


def create_constraints(variables_modalities, external_data, vec_all_incomes, grouped_pop):
    variables = list(variables_modalities.keys())

    ech = {}
    constraint = {}
    for variable in variables:
        ech[variable] = {}

        decile = external_data[external_data["modality"].isin(variables_modalities[variable])]

        for modality in variables_modalities[variable]:
            decile_tmp = decile[decile["modality"].isin([modality])]
            total_population_decile_tmp = [
                float(decile_tmp["D1"]),
                float(decile_tmp["D2"]),
                float(decile_tmp["D3"]),
                float(decile_tmp["D4"]),
                float(decile_tmp["D5"]),
                float(decile_tmp["D6"]),
                float(decile_tmp["D7"]),
                float(decile_tmp["D8"]),
                float(decile_tmp["D9"]),
                float(decile_tmp["D9"]) * DECILE_10,
            ]
            p_R_tmp = pd.DataFrame({"income": vec_all_incomes})
            p_R_tmp["proba1"] = p_R_tmp.apply(
                lambda x: utils.interpolate_income(x["income"], total_population_decile_tmp),
                axis=1,
            )
            ech[variable][modality] = p_R_tmp

        # p
        # get statistics (frequency)
        prob_1 = grouped_pop.groupby([variable], as_index=False)["probability"].sum()
        # multiply frequencies by each element of ech_compo
        for modality in ech[variable]:
            value = prob_1[prob_1[variable].isin([modality])]
            if len(value) > 0:
                probability = value["probability"]
            else:
                probability = 0
            df = ech[variable][modality]
            df["proba1"] = df["proba1"] * float(
                probability
            )  # prob(income | modality) * frequency // ech is modified inplace here

        ech_list = []
        for modality in ech[variable]:
            ech_list.append(ech[variable][modality])
        C = pd.concat(
            ech_list,
            axis=1,
        )

        C = C.iloc[:, 1::2]
        C.columns = list(range(0, len(ech[variable])))
        C["Proba"] = C.sum(axis=1)
        p = C[["Proba"]]

        # constraint
        constraint[variable] = {}
        for modality in ech[variable]:
            constraint[variable][modality] = ech[variable][modality]["proba1"] / p["Proba"]

    return constraint


# functions for creating and running model


def create_samplespace_and_features(variables_modalities, group):
    """
    Create model samplespace and features from variables and their modalities.

    :param variables_modalities: {variable: [variable modalities]}

    :return: samplespace, features
    """

    # samplespace is the set of all possible combinations
    variables = list(variables_modalities.keys())
    samplespace = list(product(*variables_modalities.values()))
    samplespace = [{variables[i]: x[i] for i in range(len(x))} for x in samplespace]

    features = []

    # base feature is x in samplespace
    def f0(x):
        return x in samplespace

    features.append(f0)

    # add a feature for all modalities except one for all variables
    for variable in variables_modalities.keys():
        for modality in variables_modalities[variable][:-1]:
            features.append(modality_feature(variable, modality))

    # create prior df
    prior_df = pd.DataFrame.from_dict(samplespace)
    prior_df_perc = prior_df.merge(group, how="left", on=variables)
    prior_df_perc["probability"] = prior_df_perc.apply(
        lambda x: 0 if x["probability"] != x["probability"] else x["probability"], axis=1
    )

    # get non zero entries
    prior_df_perc_reducted = prior_df_perc.query("probability > 0")

    # get reducted samplespace
    samplespace_reducted = prior_df_perc_reducted[variables].to_dict(orient="records")

    def function_prior_prob(x_array):
        return prior_df_perc_reducted["probability"].apply(math.log)

    return samplespace_reducted, features, function_prior_prob


def modality_feature(variable, modality):
    def feature(x):
        return x[variable] == modality

    return feature


def create_model(features, samplespace, prior_log_pdf):
    """
    Create a MinDivergenceModel instance on the given parameters.

    :param features: list of feature functions
    :param samplespace: model samplespace
    :param prior_log_pdf: prior function

    :return: MinDivergenceModel instance

    """
    model = maxentropy.MinDivergenceModel(
        features,
        samplespace,
        vectorized=False,
        verbose=0,
        prior_log_pdf=prior_log_pdf,
    )
    return model


def run_model_on_income(model_with_apriori, i, modalities, constraint):
    res = None

    try:
        K = [1]

        for variable in modalities:

            for modality in modalities[variable][:-1]:

                K.append(constraint[variable][modality][i])

        K = np.array(K).reshape(1, len(K))

        res = compute_rq(model_with_apriori, np.shape(K)[1], K)

        model_with_apriori.fit(K)

        print("SUCCESS on income " + str(i) + " with fun=" + str(res.fun))
    except (Exception, maxentropy.utils.DivergenceError) as e:
        print("ERROR on income " + str(i) + " with fun=" + str(res.fun))


# check functions


def compute_rq(model, nb_modalities, K):
    I = np.identity(nb_modalities)
    A = model.F.A
    A_eq = np.concatenate([A, I], axis=1)

    c = np.concatenate([np.zeros(np.shape(A)[1]), np.ones(nb_modalities)], axis=None)

    res = linprog(c, A_eq=A_eq, b_eq=K, method="simplex")

    return res
