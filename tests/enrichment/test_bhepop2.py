from bhepop2.enrichment.bhepop2 import Bhepop2Enrichment

import numpy as np


class TestBhepop2Enrichment:

    def test_init(
        self, synthetic_population_nantes, quantitative_marginal_distributions, test_seed
    ):
        enrich_class = Bhepop2Enrichment(
            synthetic_population_nantes,
            quantitative_marginal_distributions,
        )

        # check class attributes
        assert hasattr(enrich_class, "crossed_modalities_frequencies")
        assert hasattr(enrich_class, "crossed_modalities_matrix")
        assert hasattr(enrich_class, "constraints")
        assert hasattr(enrich_class, "optim_result")

        # check modalities property
        assert enrich_class.modalities == quantitative_marginal_distributions.modalities

    def test_bhepop2_enrichment(
        self,
        synthetic_population_nantes,
        quantitative_marginal_distributions,
        test_seed,
        expected_enriched_population_nantes,
    ):
        """
        Integration test of bhepop2 enrichment.
        """
        feature_name = "feature"
        synthetic_population_nantes = synthetic_population_nantes.drop("sex", axis=1)

        enrich_class = Bhepop2Enrichment(
            synthetic_population_nantes,
            quantitative_marginal_distributions,
            feature_name=feature_name,
            seed=test_seed,
        )

        enriched_pop = enrich_class.assign_feature_values()
        enriched_pop[feature_name] = enriched_pop[feature_name].astype(int)

        # uncomment to update test data
        # enriched_pop.to_csv("tests/data/nantes_enriched.csv", index=False)

        assert np.all((enriched_pop == expected_enriched_population_nantes).to_numpy())
