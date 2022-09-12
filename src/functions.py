from itertools import product
from scipy.optimize import linprog
import maxentropy
import pandas as pd
import numpy as np
import math


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
    print(samplespace)
    prior_df = pd.DataFrame.from_dict(samplespace)
    print(prior_df)
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

        print(K)
        print(model_with_apriori.expectations())

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
