
import logging as lg
import numpy as np
from itertools import product
import pandas as pd
from src import utils
from src import functions2
import maxentropy
import math


class MaxEntropyEnrichment:

    def __init__(self, population, distributions, commune_id, modalities=None, parameters=None):
        """
        Synthetic population enrichment class.

        :param population: enriched population
        :param distributions: enriching data distributions
        :param commune_id: spatial selection
        :param modalities: distribution modalities used. Default is all
        :param parameters: enrichment parameters
        """

        # original population to be enriched
        self.population = None

        # distributions of the feature values by modality
        self.distributions = None

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities]
        self.modalities = None

        # execution parameters TODO : validate with json schema
        self.parameters = parameters

        # commune id
        self.commune_id = commune_id

        # algorithm data

        # vector of feature values defining the intervals
        self.feature_values = None

        # total number of features
        self.nb_features = None

        # frequency of each crossed modality present in the population
        self.crossed_modalities_frequencies = None

        # maxentropy MinDivergenceModel instance
        self.maxentropy_model = None

        self.log("Initialisation of optimisation data")

        self._init_distributions(distributions, modalities)
        self._init_modalities(modalities)
        self._init_population(population)

    def _init_distributions(self, distributions, modalities):

        validate_distributions(distributions)

        distributions = distributions.copy()

        distributions = distributions.query(f"commune_id == '{self.commune_id}'")

        distributions = distributions[distributions["attribute"].isin(functions2.get_attributes(modalities))]

        self.distributions = distributions

    def _init_modalities(self, modalities):
        self.modalities = modalities
        # infer modalities from distributions if None


    def _init_population(self, population):
        validate_population(population, self.modalities)
        # population = population.query(f"commune_id == '{self.commune_id}'")
        self.population = population



    def main(self):
        # compute crossed modalities frequencies
        self.crossed_modalities_frequencies = functions2.compute_crossed_modalities_frequencies(self.population, self.modalities)

        # compute vector of feature values
        self.feature_values = functions2.compute_feature_values(self.distributions, self.parameters["abs_minimum"], self.parameters["relative_maximum"])

        self.create_maxentropy_model()

        # run resolution
        res = self.run_assignment()

        return res

    def create_maxentropy_model(self):

        # prepare data for maxentropy resolution

        samplespace_reducted, model_features_functions, function_prior_prob = self.create_samplespace_and_features()

        self.create_model(model_features_functions, samplespace_reducted, function_prior_prob)

    def run_assignment(self):

        constraints = self.compute_constraints()

        features = [0, 1, 2, 3, 4, 5, 6]
        res = pd.DataFrame()
        # loop on features
        for i in features:
            print("Running model for feature " + str(i))

            self.run_model_on_feature(self.maxentropy_model, i, self.modalities, constraints)
            res.loc[:, i] = self.maxentropy_model.probdist()

            # need to reset dual for next iterations !
            self.maxentropy_model.resetparams()

        return res

    def run_model_on_feature(self, model_with_apriori, i, modalities, constraint):
        # res = None

        try:
            K = [1]

            for variable in modalities:

                for modality in modalities[variable][:-1]:
                    K.append(constraint[variable][modality][i])

            K = np.array(K).reshape(1, len(K))

            # res = compute_rq(model_with_apriori, np.shape(K)[1], K)

            model_with_apriori.fit(K)

            # print("SUCCESS on feature " + str(i) + " with fun=" + str(res.fun))

        except (Exception, maxentropy.utils.DivergenceError) as e:
            pass
            # print("ERROR on feature " + str(i) + " with fun=" + str(res.fun))

    def create_samplespace_and_features(self):
        """
        Create model samplespace and features from variables and their modalities.

        :return: samplespace, features
        """

        # samplespace is the set of all possible combinations
        attributes = functions2.get_attributes(self.modalities)
        samplespace = list(product(*self.modalities.values()))
        samplespace = [{attributes[i]: x[i] for i in range(len(x))} for x in samplespace]

        features = []

        # base feature is x in samplespace
        def f0(x):
            return x in samplespace

        features.append(f0)

        # add a feature for all modalities except one for all variables
        for attribute in attributes:
            for modality in self.modalities[attribute][:-1]:
                features.append(functions2.modality_feature(attribute, modality))

        # create prior df
        prior_df = pd.DataFrame.from_dict(samplespace)
        prior_df_perc = prior_df.merge(self.crossed_modalities_frequencies, how="left", on=attributes)
        prior_df_perc["probability"] = prior_df_perc.apply(
            lambda x: 0 if x["probability"] != x["probability"] else x["probability"], axis=1
        )

        # get non zero entries
        prior_df_perc_reducted = prior_df_perc.query("probability > 0")

        # TODO : just use crossed modalities freq for prior
        # prior_df_perc_reducted.reset_index(inplace=True, drop=True)
        # print(prior_df_perc_reducted)
        # self.crossed_modalities_frequencies.reset_index(inplace=True, drop=True)
        # print(self.crossed_modalities_frequencies)
        # print(prior_df_perc_reducted == self.crossed_modalities_frequencies)

        # get reducted samplespace
        samplespace_reducted = prior_df_perc_reducted[attributes].to_dict(orient="records")

        def function_prior_prob(x_array):
            return prior_df_perc_reducted["probability"].apply(math.log)

        return samplespace_reducted, features, function_prior_prob

    def compute_constraints(self):
        """
        For each modality of each attribute, compute the probability of belonging to each feature interval.

        P(f in [Fi, Fi+1] | Modality) = P(f in [Fi, Fi+1] | Modality) * P(Modality) / P(f in F[i, Fi+1])

        :return:
        """

        # compute constraints on each modality
        attributes = functions2.get_attributes(self.modalities)

        ech = {}
        constraint = {}
        for attribute in attributes:
            ech[attribute] = {}
            # get attribute frequency
            attribute_freq = self.crossed_modalities_frequencies.groupby([attribute], as_index=False)["probability"].sum()
            for modality in self.modalities[attribute]:
                # compute probability of each feature interval when being in a modality
                ech[attribute][modality] = self.compute_feature_prob(attribute, modality)

                # multiply frequencies by each element of ech_compo
                value = attribute_freq[attribute_freq[attribute].isin([modality])]
                if len(value) > 0:
                    probability = value["probability"]
                else:
                    probability = 0
                df = ech[attribute][modality]
                # prob(feature | modality) * frequency // ech is modified inplace here
                df["prob"] = df["prob"] * float(
                    probability
                )

            ech_list = []
            for modality in ech[attribute]:
                ech_list.append(ech[attribute][modality])
            C = pd.concat(
                ech_list,
                axis=1,
            )
            # Somme P(feature & modality) sur les modality = P(feature)
            C = C.iloc[:, 1::2]
            C.columns = list(range(0, len(ech[attribute])))
            C["Proba"] = C.sum(axis=1)
            p = C[["Proba"]]

            # constraint
            constraint[attribute] = {}
            for modality in ech[attribute]:
                constraint[attribute][modality] = ech[attribute][modality]["prob"] / p["Proba"]

        return constraint

    def create_model(self, features, samplespace, prior_log_pdf):
        """
        Create and set a MinDivergenceModel instance on the given parameters.

        :param features: list of feature functions
        :param samplespace: model samplespace
        :param prior_log_pdf: prior function
        """

        # create and set MinDivergenceModel instance
        self.maxentropy_model = maxentropy.MinDivergenceModel(
            features,
            samplespace,
            vectorized=False,
            verbose=self.parameters["maxentropy_verbose"],
            prior_log_pdf=prior_log_pdf,
            algorithm=self.parameters["maxentropy_algorithm"],
        )

    def compute_feature_prob(self, attribute, modality):

        decile_tmp = self.distributions[self.distributions["modality"].isin([modality]) & self.distributions["attribute"].isin([attribute])]

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
            float(decile_tmp["D9"]) * self.parameters["relative_maximum"],
        ]

        p_R_tmp = pd.DataFrame({"feature": self.feature_values})
        p_R_tmp["prob"] = p_R_tmp.apply(
            lambda x: functions2.interpolate_feature_prob(x["feature"], total_population_decile_tmp),
            axis=1,
        )

        return p_R_tmp

    def compute_crossed_modality_probs(self):
        """
        For each feature interval, compute the probability of belonging to a crossed modality.

        P(Mk | f in [Fi, Fi+1])

        :return: DataFrame with K lines and N columns
        """



    def get_feature_probs(self):
        """
        For each crossed modality, compute the probability of belonging to a feature interval.

        Invert the crossed modality probabilities using Bayes.

        Compute P(F in [Fi, Fi+1] | Mk) = P(Mk | f in [Fi, Fi+1]) * P(Mk) / P(f in [Fi, Fi+1])

        :return: DataFrame
        """

    def compute_feature_probabilities_from_distributions(self):
        """
        Compute the probability of each feature interval.

        Use the global distribution of the features to interpolate the interval probabilities.

        The resulting DataFrame contains P(f in [Fi, Fi+1]) for i in N

        :return: DataFrame
        """

        pass

    # utils methods

    def log(message: str, level: int = lg.DEBUG):
        """
        Log a message using the package logger.

        See logging library.

        :param message: message to be logged
        :param level: logging level
        """

        utils.log(message, level)

    log = staticmethod(log)

def validate_distributions(distributions):
    # we could validate the distributions (positive, monotony ?)
    assert {*["D{}".format(i) for i in range(1, 10)], "attribute", "modality"} <= set(distributions.columns)

def validate_population(population, modalities):

    attributes = functions2.get_attributes(modalities)

    # { id } and commune_id mandatory ?
    assert {*attributes} <= set(population.columns)

    for attribute in attributes:
        assert population[attribute].isin(modalities[attribute]).all()
