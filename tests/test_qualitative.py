from bhepop2.enrichment.bhepop2 import Bhepop2Enrichment
from bhepop2.sources.marginal_distributions import QualitativeMarginalDistributions

import pytest
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
    table_percentage_attribute2 = (
        table_numbers.transpose().sum() / table_numbers.transpose().sum().sum()
    )
    table_percentage = table_numbers / table_numbers.sum()
    table_percentage["all"] = table_percentage_attribute2
    table_percentage = table_percentage.transpose()
    table_percentage["modality"] = table_percentage.index
    table_percentage["attribute"] = name_attribute1

    return table_percentage


@pytest.fixture(scope="session")
def pop_synt_men_nantes():
    synt_pop = pd.read_csv("tests/data/pop_synth_men_nantes.csv")
    return synt_pop


@pytest.fixture(scope="session")
def synth_pop_defected(pop_synt_men_nantes):
    return pop_synt_men_nantes.drop(["Voit_rec"], axis=1)


@pytest.fixture(scope="session")
def distributions(pop_synt_men_nantes):
    attributes = list(pop_synt_men_nantes.columns[:-1])
    marginal_distribution = pd.concat(
        list(map(lambda a: build_cross_table(pop_synt_men_nantes, [a, "Voit_rec"]), attributes))
    )
    marginal_distribution = marginal_distribution.loc[
        ~marginal_distribution.index.duplicated(keep="first")
    ]
    marginal_distribution.loc["all", "attribute"] = "all"
    return marginal_distribution


def test_qualitative(synth_pop_defected, distributions, test_seed):
    distributions = QualitativeMarginalDistributions(distributions)
    enrich_class = Bhepop2Enrichment(synth_pop_defected, distributions, seed=test_seed)
    # Run optimisation

    pop = enrich_class.assign_features()

    nb_cars_dict = pop.groupby("feature").agg("size").to_dict()
    assert nb_cars_dict["0voit"] == 89197
    assert nb_cars_dict["1voit"] == 299405
    assert nb_cars_dict["2voit"] == 207335
    assert nb_cars_dict["3voit"] == 33395
    assert sum(nb_cars_dict.values()) == len(pop)


# def test_qualitative_enrich(
#     pop_synt_men_nantes,
#     distributions,
#     test_modalities,
#     test_parameters,
#     test_seed,
#     expected_enriched_population_nantes,
# ):
#     modalities = infer_modalities_from_distributions(distributions)
#     modalities["Voit_rec"] = ["0voit", "1voit", "2voit", "3voit"]
#     test = compute_crossed_modalities_frequencies(pop_synt_men_nantes, modalities)
#
#
#     def calc(x):
#         return x["probability"] / distributions.loc["all", x["Voit_rec"]]
#
#     test["prob_cond"] = test.apply(calc, axis=1)
#
#     test.apply(
#     calc,
#     axis=1,
#     )
#
#     # le test commence ici
#     synt_pop_defected = pop_synt_men_nantes.drop(['Voit_rec'], axis=1)
#
#     enrich_class = QualitativeEnrichment(
#     synt_pop_defected, distributions, seed=test_seed
#     )
#
#     # Run optimisation
#     enrich_class.optimise()
#
#
#     # Assign data to synthetic population
#     pop = enrich_class.assign_feature_value_to_pop()
#     print(pop.head())
#     pop.rename(columns={"feature":"Voit_rec"},inplace=True) # ename feaure in Voit_rec in order to compare with the true population
#
#     # this solution will be compare with a naive resolution
#     # by samping according the general frequeny in the _init_population
#     # Récupérez les valeurs de 0voit ... 3voit pour la ligne où Voit_rec vaut 'all'
#
#     ligne_all_voit = distributions.loc[distributions['attribute'] == 'all', '0voit':'3voit'].values
#     ligne_all_voit = ligne_all_voit[0] # the output of the previous line is 2D, it is downsized to 1D
#     common_pop=synt_pop_defected
#     common_pop['Voit_rec']=np.random.choice(['0voit', '1voit', '2voit', '3voit'], size=len(common_pop), p=ligne_all_voit)
#     print(common_pop.head())
#
#     prediction = compute_crossed_modalities_frequencies(pop, modalities)
#     common_prediction = compute_crossed_modalities_frequencies(common_pop, modalities)
#
#     def build_probability_vector(probability_df, modalities, feature_column):
#         """
#         Inspecte les modalités croisées entre les colonnes du DataFrame probability_df et les modalités fournies
#         dans le dictionnaire modalities. Ajoute la valeur feature si la combinaison de modalités existe dans probability_df,
#         sinon ajoute zéro.
#
#         Args:
#         probability_df (pd.DataFrame): DataFrame à inspecter.
#         modalities (dict): Dictionnaire des modalités pour chaque colonne.
#         feature_column (str): Nom de la colonne contenant les valeurs de feature.
#
#         Returns:
#          np.array : vecteur des valeurs de feature ou zéro pour chaque combinaison de modalités.
#
#         Todo:
#           si on trouve plusieurs valeurs de probabilités pour la même modalité croisé, c'est une erreur
#         """
#         result = []
#         # Génère toutes les combinaisons possibles de modalités
#         combinations = list(itertools.product(*modalities.values()))
#
#         for combination in combinations:
#             # Vérifie si la combinaison de modalités existe dans pop
#             mask = probability_df.apply(lambda row: all(row[colname] in [modality] for colname, modality in zip(modalities.keys(), combination)), axis=1)
#             if mask.any():
#                 # Ajoute la valeur de feature si la combinaison de modalités existe
#                 result.append(probability_df.loc[mask, feature_column].values[0])
#             else:
#                 # Ajoute zéro si la combinaison de modalités n'existe pas
#                 result.append(0)
#         return np.array(result)
#
#     # Exemple d'utilisation
#     frequencies_vector = build_probability_vector(test, modalities, 'probability')
#     probabilities_vector = build_probability_vector(prediction, modalities, 'probability')
#     common_probabilities_vector = build_probability_vector(common_prediction, modalities, 'probability')
#
#     def compute_ratio(proba, freq, nagents, nmodalities, eps_proba, proba_min):
#         """
#         calcule les métriques entre les proba estimées et les fréquences réelles
#            RE : relative Error
#            R2 : coefficient of determination
#            RMSD : root Mean Square Deviation
#            MAPE : Mean Absolute Percentage Error
#            divergence : kullback leibler divergence
#            Goodness of fit :
#
#
#         Args:
#         proba : vecteur des probabilités estimés
#         freq : vecteur des fréquences réelles
#         nmodalities : nombre de modalités croisées
#         eps_proba : niveau pour détecter quand une probabilité est nulle
#         proba_min : une proba détectée comme nulle est mise à la valeur proba_min
#         eps : si v_input(k) < eps, il est considéré comme nul
#         proba_min : les probabilités nulles sont remplacés par la valeur proba_min
#
#         Returns:
#         rmsd, mape, divergence, goodness_of_fit
#
#         """
#
#         def support_vector_not_null(v_input, eps, proba_min):
#             """
#             construit un vecteur de probabilité qui ne contient pas probabilité nulle, ceci permet de comparer des vecteurs de probabilités qui n'ont pas le même support
#
#             Args:
#             v_input : vecteur de probabilité avec des composants nul
#             eps : si v_input(k) < eps, il est considéré comme nul
#             proba_min : les probabilités nulles sont remplacés par la valeur proba_min
#
#             Returns:
#             v_ouput: vecteur de probabilité dont aucune composante n'est nulles
#
#             TODO : la somme de v_output ne vaut plus 1... To discuss
#                     pas de test en entrée
#             """
#             v_output=v_input.copy()
#             v_output[v_input<eps]=proba_min
#             return v_output/np.sum(v_output)
#
#
#
#         freq_not_null = support_vector_not_null(freq,eps_proba,proba_min)
#         proba_not_null = support_vector_not_null(proba,eps_proba,proba_min)
#
#         diff_vector = proba - freq
#
#         re = np.linalg.norm(diff_vector)/np.linalg.norm(freq)
#         rmsd = np.linalg.norm(diff_vector)/np.sqrt(nmodalities)
#         r2 = 1-np.sum(diff_vector**2)/np.sum(freq**2)
#
#         diff_vector = proba - freq_not_null
#         mape = (np.sum(np.abs(diff_vector) / freq_not_null))/nmodalities
#
#         divergence = np.sum( freq_not_null*np.log(freq_not_null/proba_not_null))
#
#         diff_vector = proba_not_null - freq
#         goodness_of_fit = np.sum(diff_vector**2 / proba_not_null)
#
#         return re, r2, rmsd, mape, divergence, goodness_of_fit
#
#     nagents = len(pop)
#     nmodalities = len(frequencies_vector)
#     eps_proba = 1e-6/nagents
#     proba_min = 0.1/nagents # tobe discussed
#
#     re, r2, rmsd, map, divergence, goodness_of_fit = compute_ratio(probabilities_vector, frequencies_vector, nagents, nmodalities, eps_proba, proba_min)
#     c_re, c_r2, c_rmsd, c_map, c_divergence, c_goodness_of_fit = compute_ratio(common_probabilities_vector, frequencies_vector, nagents, nmodalities, eps_proba, proba_min)
#
#
#     # Display tests results
#     metric_bhepop2 = [re,r2,rmsd,map,divergence,goodness_of_fit]
#     metric_common = [c_re,c_r2,c_rmsd,c_map,c_divergence,c_goodness_of_fit]
#     criteria = ['Relative Error', 'Coef of determination', 'Root Mean Square Deviration', 'Mean Absolute Percentage Error', 'Divergence', 'Goodness of fit' ]
#     results_test = pd.DataFrame({'Common': metric_common, 'Bhepop2': metric_bhepop2 },index=criteria)
#     print(results_test)
#     assert 0 #POV pour rester dans l'environnement lorsque nous lançons pytest --pdb tests/test_qualitative.py -s
