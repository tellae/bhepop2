from bhepop2.quantitative_enrichment import QuantitativeEnrichment

import numpy as np


def test_bhepop2_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes,
):
    synthetic_population_nantes.drop("sex", axis=1, inplace=True)

    enrich_class = QuantitativeEnrichment(
        synthetic_population_nantes,
        filosofi_distributions_nantes,
        list(test_modalities.keys()),
        parameters=test_parameters,
        seed=test_seed,
    )

    pop = enrich_class.assign_features()
    print(pop)
    # pop.to_csv("tests/data/nantes_enriched.csv", index=False)

    assert np.all((pop == expected_enriched_population_nantes).to_numpy())
