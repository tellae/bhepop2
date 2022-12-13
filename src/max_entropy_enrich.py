
import logging as lg
import numpy as np
from itertools import product
import pandas as pd
from src import utils
from src import functions2
import maxentropy
import math
from tests.conftest import MODALITIES

class MaxEntropyEnrichment:

    def __init__(self, population: pd.DataFrame, distributions: pd.DataFrame, commune_id: str, attribute_selection: list=None, parameters=None):
        """
        Synthetic population enrichment class.

        :param population: enriched population
        :param distributions: enriching data distributions
        :param commune_id: spatial selection
        :param attribute_selection: distribution attributes used. By default, use all attributes of the distribution
        :param parameters: enrichment parameters
        """

        # original population to be enriched
        self.population = None

        # distributions of the feature values by modality
        self.distributions = None

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
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

        # optimization constraints
        self.constraints = None

        self.log("Initialisation of enrichment algorithm data", lg.INFO)

        self._init_distributions(distributions, attribute_selection)
        self._init_population(population)

    def _init_distributions(self, distributions, attribute_selection):
        """
        Validate and filter the input distributions.

        When done, set the *distributions* field.

        :param distributions: input distributions DataFrame
        :param attribute_selection: distribution attributes selection
        """

        self.log("Setup distributions data")

        # validate distributions format and contents
        functions2.validate_distributions(distributions)

        distributions = distributions.copy()

        distributions = distributions.query(f"commune_id == '{self.commune_id}'")

        # filter distributions using the attribute selection
        if attribute_selection is not None:
            distributions = distributions[distributions["attribute"].isin(attribute_selection)]
            assert set(distributions["attribute"]) == set(attribute_selection), "Mismatch between distribution attributes and attribute selection"

        # set distributions
        self.distributions = distributions

        # infer attributes and their modalities from the filtered distribution
        # self.modalities = functions2.infer_modalities_from_distributions(distributions)
        assert MODALITIES == functions2.infer_modalities_from_distributions(distributions)
        self.modalities = MODALITIES

    def _init_population(self, population):
        """
        Validate and filter the input population.

        When done, set the *population* field.

        :param population: input population DataFrame
        """
        self.log("Setup population data")
        functions2.validate_population(population, self.modalities)
        # population = population.query(f"commune_id == '{self.commune_id}'")
        self.population = population

        # TODO ? remove distributions unused by population

    def main(self):
        # compute crossed modalities frequencies
        self.log("Computing frequencies of crossed modalities", lg.INFO)
        self.crossed_modalities_frequencies = functions2.compute_crossed_modalities_frequencies(self.population, self.modalities)
        self.log("Number of crossed modalities present in the population: {}".format(len(self.crossed_modalities_frequencies)))

        # compute vector of feature values
        self.log("Computing vector of all feature values", lg.INFO)
        self.feature_values = functions2.compute_feature_values(self.distributions, self.parameters["abs_minimum"], self.parameters["relative_maximum"])
        self.log("Number of feature values: {}".format(len(self.feature_values)))

        # create and set the maxentropy model
        self.log("Creating optimization model", lg.INFO)
        self._create_maxentropy_model()

        # compute matrix of constraints
        self.log("Computing optimization constraints", lg.INFO)
        self._compute_constraints()

        # run resolution
        self.log("Starting optimization by entropy maximisation", lg.INFO)
        res = self._run_optimization()

        return res

    def _run_optimization(self) -> pd.DataFrame:
        """
        Run optimization model on each feature value.

        The resulting probabilities are the P(Mk | f in [fi, fi+1]).

        :return: DataFrame containing the result probabilities
        """

        # this is for testing, change to self.feature_values
        features = [0, 1, 2, 3, 4, 5, 6]
        res = pd.DataFrame()

        # loop on features
        for i in features:
            # run optimization model on the current feature
            self.log("Running optimization model on feature " + str(i))
            self.run_model_on_feature(i)

            # store result in DataFrame
            res.loc[:, i] = self.maxentropy_model.probdist()

            # reset dual for next iterations
            self.maxentropy_model.resetparams()

        return res

    def run_model_on_feature(self, i):
        """
        Run the optimization model on the feature of index i.

        Nothing is returned, but the result can be obtained on the model,
        for instance with self.maxentropy_model.probdist().

        :param i: index of optimized feature
        """
        # res = None

        try:
            K = [1]
            for attribute in self.modalities:
                for modality in self.modalities[attribute][:-1]:
                    K.append(self.constraints[attribute][modality][i])

            K = np.array(K).reshape(1, len(K))

            # res = compute_rq(model_with_apriori, np.shape(K)[1], K)

            self.maxentropy_model.fit(K)

        except (Exception, maxentropy.utils.DivergenceError) as e:
            self.log("Error while running optimization model on feature " + str(i), lg.ERROR)
            raise e


    def create_samplespace_and_features(self):
        """
        Create model samplespace and features from variables and their modalities.

        features: list of feature functions
        samplespace: model samplespace
        prior_log_pdf: prior function

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

    def _compute_constraints(self):
        """
        For each modality of each attribute, compute the probability of belonging to each feature interval.

        P(f in [Fi, Fi+1] | Modality) = P(f in [Fi, Fi+1] | Modality) * P(Modality) / P(f in F[i, Fi+1])

        :return:
        """

        # compute constraints on each modality
        attributes = functions2.get_attributes(self.modalities)

        ech = {}
        constraints = {}
        for attribute in attributes:
            ech[attribute] = {}
            # get attribute frequency
            attribute_freq = self.crossed_modalities_frequencies.groupby([attribute], as_index=False)["probability"].sum()
            for modality in self.modalities[attribute]:
                self.log("Computing constraints for modality: {} = {}".format(attribute, modality))

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
            constraints[attribute] = {}
            for modality in ech[attribute]:
                constraints[attribute][modality] = ech[attribute][modality]["prob"] / p["Proba"]

        self.constraints = constraints

    def _create_maxentropy_model(self):
        """
        Create and set a MinDivergenceModel instance on the given parameters.
        """

        # create data structures describing the optimization problem
        samplespace, model_features_functions, prior_log_pdf = self.create_samplespace_and_features()

        # create and set MinDivergenceModel instance
        self.maxentropy_model = maxentropy.MinDivergenceModel(
            model_features_functions,
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
