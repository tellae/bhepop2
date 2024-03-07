from bhepop2.sources.global_distribution import QuantitativeGlobalDistribution

import pytest


class TestQuantitativeGlobalDistribution:

    def test_init(self, filosofi_global_distribution_nantes, test_parameters):
        test_name = "test_name"
        abs_min = 0
        relative_max = 1.5

        global_distribution = QuantitativeGlobalDistribution(
            filosofi_global_distribution_nantes, name=test_name, abs_minimum=abs_min, relative_maximum=relative_max
        )

        # check class attributes
        assert global_distribution.data is filosofi_global_distribution_nantes
        assert global_distribution.name == test_name
        assert global_distribution._abs_minimum == abs_min
        assert global_distribution._relative_maximum == relative_max

    def test_validate_data(self, filosofi_global_distribution_nantes):

        # DataFrame with missing decile should raise an error
        with pytest.raises(AssertionError):
            QuantitativeGlobalDistribution(
                filosofi_global_distribution_nantes.drop(["D5"], axis=1)
            )

        # empty DataFrame should raise an error
        with pytest.raises(AssertionError):
            QuantitativeGlobalDistribution(
                filosofi_global_distribution_nantes.drop([filosofi_global_distribution_nantes.index[0]], axis=0)
            )
