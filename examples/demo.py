#!/usr/bin/env python
# coding: utf-8

# # Adding income information to a synthetic population of households
# 
# In this example, we already have a synthetic population of households on Nantes city. This synthetic population was built using the French national Census of the population. For each household, serveral characteristics have been added:
# 
# - Ownership : owner or tenant of its accomodation
# - Age : age of reference person
# - Size : number of persons
# - Family composition : composition (single person, couple with ou without children, etc)
# 
# The objectif is to add income information to each household. In order to reach this goal, we use another data source named Filosofi. More precisely, this data source gives information on the income distribution (deciles) for each city, per household characteristics.
# 
# Filosofi is an indicator set implemented by INSEE which is the French National Institute of Statistics. See [insee.fr](https://www.insee.fr/fr/metadonnees/source/serie/s1172) for more details.
# 

# In[1]:

import pdb # pov debug

import warnings
import pandas as pd
#from  bhepop2.max_entropy_enrich_gradient import MaxEntropyEnrichment_gradient
from  bhepop2.tools import read_filosofi, compute_distribution, plot_analysis

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("mode.chained_assignment", None)


import logging as lg
import numpy as np
import random
import pandas as pd
from bhepop2 import utils
from bhepop2 import functions
import maxentropy
import math

# POV  function and libraries used in  function minxent_gradient()
from numpy.linalg import norm 
import time

class MaxEntropyEnrichment_gradient:
    """
    A class for enriching population using entropy maximisation.

    Notations used in this class documentation:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)
    - :math:`F_{i}` : feature value i
    """

    #: json schema of the enrichment parameters
    parameters_schema = {
        "title": "MaxEntropyEnrichment parameters",
        "description": "Parameters of a population enrichment run",
        "type": "object",
        "required": [
            "abs_minimum",
            "relative_maximum",
            "maxentropy_algorithm",
            "maxentropy_verbose",
        ],
        "properties": {
            "abs_minimum": {
                "title": "Distributions absolute minimum",
                "description": "Minimum value of the feature distributions. This value is absolute, and thus equal for all distributions.",
                "type": "number",
                "default": 0,
            },
            "relative_maximum": {
                "title": "Distributions relative maximum",
                "description": "Maximum value of the feature distributions. This value is relative and will be multiplied to the last value of each distribution.",
                "type": "number",
                "default": 1.5,
                "minimum": 1,
            },
            "delta_min": {
                "title": "Minimum feature value delta",
                "description": "Minimum size of the feature intervals",
                "type": ["null", "number"],
                "default": None,
                "minimum": 0,
            },
            "maxentropy_algorithm": {
                "title": "maxentropy algorithm parameter",
                "description": "Algorithm used for maxentropy optimization. See maxentropy BaseModel class for more information.",
                "type": "string",
                "default": "Nelder-Mead",
            },
            "maxentropy_verbose": {
                "title": "maxentropy verbose parameter",
                "description": "Verbosity of maxentropy library. Set to 1 for detailed output.",
                "enum": [0, 1],
                "default": 0,
            },
        },
    }

    def __init__(
        self,
        population: pd.DataFrame,
        distributions: pd.DataFrame,
        attribute_selection: list = None,
        parameters=None,
        seed=None,
    ):
        """
        Synthetic population enrichment class.

        :param population: enriched population
        :param distributions: enriching data distributions
        :param attribute_selection: distribution attributes used. By default, use all attributes of the distribution
        :param parameters: enrichment parameters
        :param seed: random seed
        """
        
        # POV
        self.test = 0

        # random seed (maybe use a random generator instead)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # original population to be enriched
        self.population = None

        # distributions of the feature values by modality
        self.distributions = None

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
        self.modalities = None

        # execution parameters
        self.parameters = utils.add_defaults_and_validate_against_schema(
            parameters, self.parameters_schema
        )

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

        # optimization result
        self.optim_result = None

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
        functions.validate_distributions(distributions)

        distributions = distributions.copy()

        # filter distributions using the attribute selection
        if attribute_selection is not None:
            distributions = distributions[
                distributions["attribute"].isin(attribute_selection + ["all"])
            ]
            assert set(distributions["attribute"]) == set(
                attribute_selection + ["all"]
            ), "Mismatch between distribution attributes and attribute selection"

        # set distributions
        self.distributions = distributions

        # infer attributes and their modalities from the filtered distribution
        self.modalities = functions.infer_modalities_from_distributions(distributions)

    def _init_population(self, population):
        """
        Validate and filter the input population.

        When done, set the *population* field.

        :param population: input population DataFrame
        """
        self.log("Setup population data")

        functions.validate_population(population, self.modalities)

        self.population = population.copy()

        # TODO ? remove distributions unused by population

    def optimise(self):
        # compute crossed modalities frequencies
        self.log("Computing frequencies of crossed modalities", lg.INFO)
        self.crossed_modalities_frequencies = functions.compute_crossed_modalities_frequencies(
            self.population, self.modalities
        )
        self.log(
            "Number of crossed modalities present in the population: {}".format(
                len(self.crossed_modalities_frequencies)
            )
        )

        # compute vector of feature values
        self.log("Computing vector of all feature values", lg.INFO)
        self.feature_values = functions.compute_feature_values(
            self.distributions, self.parameters["relative_maximum"], self.parameters["delta_min"]
        )
        self.nb_features = len(self.feature_values)
        self.log("Number of feature values: {}".format(self.nb_features))

        # create and set the maxentropy model
        self.log("Creating optimization model", lg.INFO)
        self._create_maxentropy_model()

        # compute matrix of constraints
        self.log("Computing optimization constraints", lg.INFO)
        self._compute_constraints()

        # run resolution
        self.log("Starting optimization by entropy maximisation", lg.INFO)
        self.optim_result = self._run_optimization()

        return self.optim_result

    def _run_optimization(self) -> pd.DataFrame:
        """
        Run optimization model on each feature value.

        The resulting probabilities are the :math:`P(M_{k} \mid f < F_{i})`.

        :return: DataFrame containing the result probabilities
        """

        res = pd.DataFrame()

        # loop on features
        for i in range(self.nb_features):
            # run optimization model on the current feature POV
            #self.log("Running optimization model on feature " + str(i))
            self.log("Running GRADIENT optimization model on feature " + str(i))
            self._run_model_on_feature(i)

            # store result in DataFrame
            # POV
            #res.loc[:, i] = self.maxentropy_model.probdist() POV mettre en place un autre monde :-)
            size_constrains_matrix=self.maxentropy_model.F.shape
            nl=size_constrains_matrix[0] # numbrt of lines i.e. number of lagrangians
            sparse_matrix=self.maxentropy_model.F.copy()
            dense_matrix=sparse_matrix.toarray()
            q=self.crossed_modalities_frequencies["probability"].values.copy()
            eta=self.maxentropy_model.K.copy()
            #q=q.flatten()
            #q.shape=(q.shape[0],1)
            #eta=eta.flatten()
            #eta.shape=(eta.shape[0],1)
            lambda_=np.zeros(nl-1)
            maxiter=[1000]
            res.loc[:, i] = self.minxent_gradient(q=q,G=dense_matrix,eta=eta,lambda_=lambda_,maxiter=maxiter) # POV
            self.test=1

            # reset dual for next iterations
            self.maxentropy_model.resetparams()

        return res

    def _run_model_on_feature(self, i):
        """
        Run the optimization model on the feature of index i.

        Nothing is returned, but the result can be obtained on the model,
        for instance with self.maxentropy_model.probdist().

        The computed probs are the :math:`P(M_{k} \mid f < F_{i})`

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

    def _create_samplespace_and_features(self):
        """
        Create model samplespace and features from variables and their modalities.

        features: list of feature functions
        samplespace: model samplespace
        prior_log_pdf: prior function

        :return: samplespace, features, prior function
        """

        # samplespace is the set of all possible combinations
        attributes = functions.get_attributes(self.modalities)

        # get samplespace
        samplespace_reducted = self.crossed_modalities_frequencies[attributes].to_dict(
            orient="records"
        )

        features = []

        # base feature is x in samplespace
        def f0(x):
            return x in samplespace_reducted

        features.append(f0)

        # add a feature for all modalities except one for all variables
        for attribute in attributes:
            for modality in self.modalities[attribute][:-1]:
                features.append(functions.modality_feature(attribute, modality))

        def function_prior_prob(x_array):
            return self.crossed_modalities_frequencies["probability"].apply(math.log)

        return samplespace_reducted, features, function_prior_prob

    def _compute_constraints(self):
        """
        For each modality of each attribute, compute the probability of belonging to each feature interval.

        .. math::
            P(Modality \mid f < F_{i}) = P(f < F_{i} \mid Modality) \\cdot \\frac{P(Modality)}{P(f < F_{i})}
        """

        # compute constraints on each modality
        attributes = functions.get_attributes(self.modalities)

        ech = {}
        constraints = {}
        for attribute in attributes:
            ech[attribute] = {}
            # get attribute frequency
            attribute_freq = self.crossed_modalities_frequencies.groupby(
                [attribute], as_index=False
            )["probability"].sum()
            for modality in self.modalities[attribute]:
                self.log("Computing constraints for modality: {} = {}".format(attribute, modality))

                # compute probability of each feature interval when being in a modality
                ech[attribute][modality] = self._compute_feature_prob(attribute, modality)

                # multiply frequencies by each element of ech_compo
                value = attribute_freq[attribute_freq[attribute].isin([modality])]
                if len(value) > 0:
                    probability = value["probability"]
                else:
                    probability = 0
                df = ech[attribute][modality]
                # prob(feature | modality) * frequency // ech is modified inplace here
                df["prob"] = df["prob"] * float(probability)

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
        (
            samplespace,
            model_features_functions,
            prior_log_pdf,
        ) = self._create_samplespace_and_features()

        # create and set MinDivergenceModel instance
        self.maxentropy_model = maxentropy.MinDivergenceModel(
            model_features_functions,
            samplespace,
            vectorized=False,
            verbose=self.parameters["maxentropy_verbose"],
            prior_log_pdf=prior_log_pdf,
            algorithm=self.parameters["maxentropy_algorithm"],
        )

    def _compute_feature_prob(self, attribute, modality):
        """
        Compute the probability of being in each feature interval with the given modality.

        :param attribute: attribute name
        :param modality: attribute modality

        :return: DataFrame
        """

        decile_tmp = self.distributions[
            self.distributions["modality"].isin([modality])
            & self.distributions["attribute"].isin([attribute])
        ]

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

        prob_df = functions.compute_features_prob(self.feature_values, total_population_decile_tmp)

        return prob_df

    def _get_feature_probs(self):
        """
        For each crossed modality, compute the probability of belonging to a feature interval.

        Invert the crossed modality probabilities using Bayes.

        Compute

        .. math::

            P(f < F_{i} \mid M_{k}) = P(M_{k} \mid f < F_{i}) \\cdot \\frac{P(f < F_{i})}{P(M_{k})}

        :return: DataFrame
        """

        feature_probs = self._compute_feature_probabilities_from_distributions() 

        res = self.optim_result 
        
        nb_columns = len(res.columns)

        for c in res.columns:
            res[c] = res[c].map(lambda x : x[0]) # POV sur conseils de Valentin
            
        
        for i in range(nb_columns):
           res[i] = res[i] * feature_probs["prob"][i]
        

        for i in range(len(res)):
            last = res.iloc[i, -1]
            res.iloc[i, :] = res.iloc[i, :] / last
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.width", 1000)

        cumulated_results = res.to_numpy()

        matrix = []

        for l in range(len(res)):
            last = cumulated_results[l, 0]
            row = [last]
            for i in range(1, self.nb_features):
                prob = cumulated_results[l, i] - last
                if prob <= 0:
                    row.append(np.nan)
                else:
                    row.append(prob)

                last = cumulated_results[l, i]

            matrix.append(row)

        probs = pd.DataFrame(matrix, columns=res.columns)

        return probs

    def assign_feature_value_to_pop(self):
        """
        Assign feature values to the population individuals using the algorithm results.
        

        :return: enriched population DataFrame
        """

        self.log("Drawing feature values for the population", lg.INFO)

        # compute the probability of being in each feature interval, for each crossed modality
        res = self._get_feature_probs()

        # associate each individual to a crossed modality
        self.crossed_modalities_frequencies["index"] = self.crossed_modalities_frequencies.index
        merge = self.population.merge(
            self.crossed_modalities_frequencies,
            how="left",
            on=functions.get_attributes(self.modalities),
        )

        # associate a feature value to the population individuals
        merge["feature"] = merge["index"].apply(lambda x: self._draw_feature(res, x))

        # remove irrelevant columns
        merge.drop(["index", "probability"], axis=1, inplace=True)

        return merge

    def _draw_feature(self, res, index):
        """
        Draw a feature value using the given distribution.

        :param res: feature distributions by crossed modality
        :param index: index of the crossed modality

        :return: Drawn feature value
        """

        # get probs
        probs = res.loc[index,].to_numpy()
        interval_values = [self.parameters["abs_minimum"]] + self.feature_values

        # get the non-null probs and values
        probs2 = []
        values2 = []
        for i in range(len(probs)):
            if pd.isna(probs[i]):
                continue
            probs2.append(probs[i])
            values2.append(interval_values[i])

        # TODO : probs don't sum to 1, this isn't reassuring

        # draw a feature interval using the probs
        values = list(range(len(values2)))
        feature_interval = random.choices(values, probs2)[0]

        # draw a feature value using a uniform distribution in the interval
        lower, upper = interval_values[feature_interval], interval_values[feature_interval + 1]
        draw = random.random()
        final = lower + round((upper - lower) * draw) #POV : pourquoi on arrondit ? Cela n'explique pas les problèmes


        return final

    def _compute_feature_probabilities_from_distributions(self):
        """
        Compute the probability of each feature interval.

        Use the global distribution of the features to interpolate the interval probabilities.

        The resulting DataFrame contains :math:`P(f < F_{i})` for i in N

        :return: DataFrame
        """
        distrib_all_df = self.distributions[self.distributions["attribute"] == "all"]

        assert len(distrib_all_df) == 1

        total_population_decile = [
            self.distributions["D1"].iloc[0],
            self.distributions["D2"].iloc[0],
            self.distributions["D3"].iloc[0],
            self.distributions["D4"].iloc[0],
            self.distributions["D5"].iloc[0],
            self.distributions["D6"].iloc[0],
            self.distributions["D7"].iloc[0],
            self.distributions["D8"].iloc[0],
            self.distributions["D9"].iloc[0],
            self.feature_values[-1],  # maximum value of all deciles
        ]

        prob_df = functions.compute_features_prob(self.feature_values, total_population_decile)

        return prob_df

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
    
    

    def minxent_gradient(q: np.ndarray, G: np.ndarray, eta: np.ndarray, lambda_: np.ndarray, maxiter: list):
        
        startTime = time.time()
        q=q.astype(float)
        G=G.astype(float)
        eta=eta.astype(float)
        lambda_=lambda_.astype(float)
        G_full=G.copy()
        eta_full=eta.copy()
        G=G_full[1:,:]
        eta=eta_full[1:]
        dimG=G.shape
        ncons=dimG[0] # POV number of constrains wihout the natural constraints
        nproba=dimG[1] # POV number of probabilites to find
        q=q.reshape(nproba,1)
        lambda_=lambda_.reshape(ncons,1)
        eta=eta.reshape(ncons,1)
        
        n_stop_iter = len(maxiter)
        max_iter_general = maxiter[n_stop_iter - 1]
        iter_general = 0
        Lagrangians = []
        Estimates = []
        Duration = np.zeros(n_stop_iter)
        #pdb.set_trace() # POV-debug

        
        compteur_max_iter = 1
        common_ratio_descending = 1/2
        common_ratio_ascending = 2.0
        #lambda0 = np.log(q.T.dot(np.exp(-G.T.dot(lambda_))))
        
        while True:
            iter_general += 1
             
            #lambda0 = np.log(np.sum(q * np.exp(-lambda_old.dot(G[1:, :])))) Exp_neg_Gt_lambdaPOV
            Exp_neg_Gt_lambda = np.exp(-G.T.dot(lambda_)) # POV exp(-Gt.lambda)
            lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
            pk =  (q*Exp_neg_Gt_lambda)/(q.T.dot(Exp_neg_Gt_lambda)) 
            level_objective =  lambda0 + np.sum(lambda_ * eta)
            f_old = eta-G.dot(pk)

            #f_old = fk(q=q, G=G, eta=eta, lambda_=lambda_old)


            #dev_ent = np.dot(q * np.exp(-lambda0) * np.exp(-lambda_old.dot(G[1:, :])), G)
            #pg = q * np.exp(-lambda0) * np.exp(-lambda_old.dot(G[1:, :]))


            alpha_ascent = 1.0
            alpha_descent = 1.0
            alpha = 1.0
            alpha_old = alpha
            lambda_old = lambda_.copy()
            lambda_ = lambda_.copy()
            lambda_new = lambda_.copy()
            level_objective_new = level_objective
            test_descent = 0
            test_ascent = 0
            
            while True:
                lambda_new = lambda_old - alpha * f_old
                lambda0 = np.log(q.T.dot(np.exp(-G.T.dot(lambda_new))))
                level_objective_new =  lambda0 + np.sum(lambda_new*eta)
                
                #level_objective_new = objective(q=q, G=G, eta=eta, lambda_=lambda_new)
                if level_objective_new > level_objective:
                    alpha_descent *= common_ratio_descending
                    alpha_old = alpha
                    alpha = alpha_descent
                    test_descent = 1
                    
                else:
                    level_objective = level_objective_new
                    lambda_ = lambda_new.copy()
                    alpha_ascent *= common_ratio_ascending
                    alpha_old = alpha
                    alpha = alpha_ascent
                    test_ascent = 1
                    
                    
                #pdb.set_trace() # POV-debug
                if test_descent * test_ascent > 0.5:
                    break
                
                if alpha < 1e-06:
                    break
            
                if alpha > 1e2:
                    break

            if  norm(lambda_ - lambda_old) < 1e-08:
                break
            
            if iter_general > max_iter_general:
                break
            
            #if iter_general == maxiter[compteur_max_iter - 1]:
             #   break

        endTime = time.time()
        duration_iter = endTime - startTime
                #lambda0 = np.log(np.sum(q * np.exp(-lambda_.dot(G[1:, :]))))
        
        Exp_neg_Gt_lambda = np.exp(-G.T.dot(lambda_)) # POV exp(-Gt.lambda)
        lambda0 = np.log(q.T.dot(Exp_neg_Gt_lambda))
        pk =  (q*Exp_neg_Gt_lambda)/(q.T.dot(Exp_neg_Gt_lambda)) 
            #pi_solve = q * np.exp(-lambda0) * np.exp(-lambda_.dot(G[1:, :]))
        Lagrangians.append([lambda0, lambda_.tolist()])
        test_pierreolivier = G.dot(pk)-eta
        print(norm(test_pierreolivier)/norm(eta))
        print(np.sum(pk))
        return pk.tolist() 
    minxent_gradient = staticmethod(minxent_gradient)

    
    
    




# ## Data preparation
# 
# Read synthetic population which doesn't contain revenu information.

# In[2]:


synth_pop = pd.read_csv("../data/inputs/nantes_synth_pop.csv", sep=";")

synth_pop.head()


# Read Filosofi data and format dataframe.

# In[3]:


filosofi = read_filosofi(
    "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
).query(f"commune_id == '44109'")

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

filosofi.head()


# ## Run algorithm

# In[4]:


# Household modalities
MODALITIES = {
    "ownership": ["Owner", "Tenant"],
    "age": ["0_29", "30_39", "40_49", "50_59", "60_74", "75_or_more"],
    "size": ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"],
    "family_comp": [
        "Single_man",
        "Single_wom",
        "Couple_without_child",
        "Couple_with_child",
        "Single_parent",
        "complex_hh",
    ],
}

# Algorithm parameters
PARAMETERS = {
    "abs_minimum": 0,
    "relative_maximum": 1.2,
    "maxentropy_algorithm": "Nelder-Mead",
    "maxentropy_verbose": 0,
    "delta_min": 1000,
}

# Optimisation preparation
enrich_class = MaxEntropyEnrichment_gradient(
    synth_pop, filosofi, list(MODALITIES.keys()), parameters=PARAMETERS, seed=42
)

# Run optimisation
enrich_class.optimise()

# Assign data to synthetic population
pop = enrich_class.assign_feature_value_to_pop()

pop.head()


# ## Results analysis

# ### Data preparation

# Format Filosofi data for comparison.

# In[5]:


filosofi_formated = filosofi.copy()
del filosofi_formated["commune_id"]
del filosofi_formated["reference_median"]

filosofi_formated = filosofi_formated.melt(
    id_vars=["attribute", "modality"],
    value_vars=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    value_name="feature",
    var_name="decile",
)
filosofi_formated["source"] = "Filosofi"


# Format simulation data for comparison.

# In[6]:


# distribution of all households
df_analysis = compute_distribution(pop)
df_analysis["attribute"] = "all"
df_analysis["modality"] = "all"

# distribution of each modality
for attribute in MODALITIES.keys():
    for modality in MODALITIES[attribute]:
        distribution = compute_distribution(pop[pop[attribute] == modality])
        distribution["attribute"] = attribute
        distribution["modality"] = modality

        df_analysis = pd.concat([df_analysis, distribution])

df_analysis["source"] = "bhepop2"


# Merge observed Filosofi and simulation data.

# In[7]:


# add filosofi data
df_analysis = pd.concat([df_analysis, filosofi_formated])

# format data
df_analysis = df_analysis.pivot(
    columns="source", index=["attribute", "modality", "decile"]
).reset_index()
df_analysis.columns = ["attribute", "modality", "decile", "Filosofi", "bhepop2"]


# ### Some plots

# In[8]:


from IPython.display import Image


# In[9]:


Image(plot_analysis(df_analysis, "all", "all").to_image())


# In[10]:


Image(plot_analysis(df_analysis, "ownership", "Tenant").to_image())


# In[11]:


Image(plot_analysis(df_analysis, "age", "30_39").to_image())


# In[12]:


Image(plot_analysis(df_analysis, "size", "3_pers").to_image())


# In[13]:


Image(plot_analysis(df_analysis, "family_comp", "Single_parent").to_image())


# %%
