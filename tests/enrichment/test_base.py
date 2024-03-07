from bhepop2.enrichment import Bhepop2Enrichment

import pytest


class TestSyntheticPopulationEnrichment:

    @pytest.fixture(scope="function")
    def enrich_class_test_instance(
        self,
        synthetic_population_nantes,
        quantitative_marginal_distributions,
        test_feature_name,
        test_seed,
    ):
        return Bhepop2Enrichment(
            synthetic_population_nantes,
            quantitative_marginal_distributions,
            feature_name=test_feature_name,
            seed=test_seed,
        )

    def test_init(
        self,
        synthetic_population_nantes,
        quantitative_marginal_distributions,
        test_seed,
        test_feature_name,
        mocker,
    ):
        # mock validation
        mocker.patch("bhepop2.enrichment.bhepop2.Bhepop2Enrichment._validate_and_process_inputs")

        enrich_class = Bhepop2Enrichment(
            synthetic_population_nantes,
            quantitative_marginal_distributions,
            feature_name=test_feature_name,
            seed=test_seed,
        )

        # check class attributes
        assert enrich_class.population.equals(synthetic_population_nantes)
        assert enrich_class.source is quantitative_marginal_distributions
        assert enrich_class.feature_name == test_feature_name
        assert enrich_class.seed == test_seed
        assert hasattr(enrich_class, "rng")
        # check validation call was made
        enrich_class._validate_and_process_inputs.assert_called_once_with()

    def test_feature_exists_error(
        self, synthetic_population_nantes, quantitative_marginal_distributions
    ):
        # feature name that already exists in population
        feature_name = "sex"

        with pytest.raises(ValueError):
            Bhepop2Enrichment(
                synthetic_population_nantes,
                quantitative_marginal_distributions,
                feature_name=feature_name,
            )

    def test_assign_feature_values(
        self, enrich_class_test_instance, test_feature_name, synthetic_population_nantes, mocker
    ):
        # mock Bhepop2 feature evaluation
        mocker.patch(
            "bhepop2.enrichment.bhepop2.Bhepop2Enrichment._evaluate_feature_on_population",
            return_value=0,
        )

        population = enrich_class_test_instance.assign_feature_values()

        # feature values are assigned to the correct column
        assert (population[test_feature_name] == 0).all()
        # the test of the population is unchanged
        population.drop([test_feature_name], axis=1, inplace=True)
        assert population.equals(synthetic_population_nantes)

    def test_compare_with_source(self, enrich_class_test_instance, test_feature_name, mocker):
        enriched_name = "test_enrich"

        # replace population by simpler item
        enrich_class_test_instance.population = "population"
        # mock Bhepop2 feature evaluation
        mocker.patch(
            "bhepop2.sources.marginal_distributions.QuantitativeMarginalDistributions.compare_with_populations",
            return_value=0,
        )

        analysis_instance = enrich_class_test_instance.compare_with_source(enriched_name, arg="arg")

        assert analysis_instance == 0
        enrich_class_test_instance.source.compare_with_populations.assert_called_once_with(
            {enriched_name: enrich_class_test_instance.population}, test_feature_name, arg="arg"
        )
