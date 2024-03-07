from bhepop2.sources.global_distribution import (
    QuantitativeGlobalDistribution,
    QuantitativeAttributes,
)

import pytest


class TestEnrichmentSource:

    def test_init(self, filosofi_global_distribution_nantes, mocker):
        test_name = "test_name"

        # mock feature values evaluation and validation
        mocker.patch(
            "bhepop2.sources.global_distribution.QuantitativeGlobalDistribution._validate_data"
        )
        mocker.patch(
            "bhepop2.sources.global_distribution.QuantitativeGlobalDistribution._evaluate_feature_values",
            return_value=[1, 2, 3],
        )

        source = QuantitativeGlobalDistribution(filosofi_global_distribution_nantes, name=test_name)

        assert source.name == test_name
        assert source.data is filosofi_global_distribution_nantes
        # check feature values
        assert source.feature_values == [1, 2, 3]
        assert source.nb_feature_values == 3
        # check validation call was made
        source._validate_data.assert_called_once_with()


class TestQuantitativeAttributes:

    def test_init(self):

        attrs = QuantitativeAttributes(0, 1.5, 1000)

        assert attrs._abs_minimum == 0
        assert attrs._relative_maximum == 1.5
        assert attrs._delta_min == 1000

    def test_relative_maximum_check(self):

        with pytest.raises(AssertionError):
            QuantitativeAttributes(relative_maximum=0.5)

    def test_delta_min_check(self):
        with pytest.raises(AssertionError):
            QuantitativeAttributes(delta_min=None)
            QuantitativeAttributes(delta_min=-1)
