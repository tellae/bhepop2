from bhepop2.max_entropy_enrich import MaxEntropyEnrichment

import numpy as np
import pandas as pd


# def test_max_entropy_enrich(
#     synthetic_population_nantes,
#     filosofi_distributions_nantes,
#     test_modalities,
#     test_parameters,
#     test_seed,
# ):
#     enrich_class = MaxEntropyEnrichment(
#         synthetic_population_nantes,
#         filosofi_distributions_nantes,
#         list(test_modalities.keys()),
#         parameters=test_parameters,
#         seed=test_seed,
#     )
#
#     enrich_class.optimise()
#
#     pop = enrich_class.assign_feature_value_to_pop()
#
#     expected_enriched_pop = pd.read_csv("tests/data/nantes_enriched_maxentropy.csv")
#
#     assert np.all((pop == expected_enriched_pop).to_numpy())
