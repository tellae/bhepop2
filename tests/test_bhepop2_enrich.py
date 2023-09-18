from bhepop2.bhepop2_enrichment import Bhepop2Enrichment

import numpy as np


def test_bhepop2_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes,
):
    enrich_class = Bhepop2Enrichment(
        synthetic_population_nantes,
        filosofi_distributions_nantes,
        list(test_modalities.keys()),
        parameters=test_parameters,
        seed=test_seed,
    )

    enrich_class.optimise()

    pop = enrich_class.assign_feature_value_to_pop()

    # pop.to_csv("tests/data/nantes_enriched.csv", index=False)

    # pop = pop.to_numpy()
    # expected_enriched_population_nantes = expected_enriched_population_nantes.to_numpy()
    # shape = np.shape(pop)
    # assert np.shape(expected_enriched_population_nantes) == shape
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         if pop[i, j] != expected_enriched_population_nantes[i, j]:
    #             print(i, j)
    #             print(pop[i, j], expected_enriched_population_nantes[i, j])
    #

    assert np.all((pop == expected_enriched_population_nantes).to_numpy())
