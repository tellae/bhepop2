from bhepop2.enrichment import SimpleUniformEnrichment
from bhepop2.sources.global_distribution import QuantitativeGlobalDistribution


class TestSimpleUniformEnrichment:

    def test_uniform_enrichment(
        self,
        synthetic_population_nantes,
        filosofi_global_distribution_nantes,
        test_feature_name,
        test_seed,
    ):
        """
        Integration test of uniform enrichment.
        """
        global_distribution = QuantitativeGlobalDistribution(
            filosofi_global_distribution_nantes, abs_minimum=0, relative_maximum=1.2
        )
        enrich_class = SimpleUniformEnrichment(
            synthetic_population_nantes,
            global_distribution,
            feature_name=test_feature_name,
            seed=test_seed,
        )

        pop = enrich_class.assign_feature_values()

        # check aggregated values on added feature
        assert round(pop[test_feature_name].agg("mean"), 2) == 22984.96
        assert round(pop[test_feature_name].agg("median"), 2) == 21225.82
