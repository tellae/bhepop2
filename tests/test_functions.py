from bhepop2.functions import *
from bhepop2.tools import compute_distribution
import numpy as np

import pytest


def test_get_attributes(test_modalities):
    """
    Test that the returned attributes are correct.
    """

    assert get_attributes(test_modalities) == ["ownership", "age", "size", "family_comp"]


def test_modality_feature():
    """
    Test that the result is a function with expected behaviour.
    """

    attribute = "my_attribute"
    modality_0 = "modality_0"
    modality_1 = "modality_1"

    feature = modality_feature(attribute, modality_0)

    assert callable(feature)
    assert feature({attribute: modality_0})
    assert not feature({attribute: modality_1})


def test_compute_crossed_modalities_frequencies(
    synthetic_population_nantes, test_modalities, test_attributes
):
    """
    Test the crossed modalities frequencies have the correct columns and sum to 1.
    """
    freq_df = compute_crossed_modalities_frequencies(synthetic_population_nantes, test_modalities)

    assert list(freq_df.columns) == test_attributes + ["probability"]
    assert np.isclose(freq_df["probability"].sum(), 1)


def test_infer_modalities_from_distributions(filosofi_distributions_nantes):
    modalities = infer_modalities_from_distributions(filosofi_distributions_nantes)

    assert isinstance(modalities, dict)
    assert "all" not in modalities


@pytest.mark.parametrize(
    "delta_min, expected_length",
    [
        (None, 190),
        (1000, 41),
    ],
)
def test_compute_feature_values(
    delta_min, expected_length, filosofi_distributions_nantes, test_attributes
):
    """
    Test that the feature values list have the correct length and is sorted.
    """

    filo = filosofi_distributions_nantes[
        filosofi_distributions_nantes["attribute"].isin(test_attributes)
    ]
    feature_values = compute_feature_values(filo, 1.5, delta_min)

    assert len(feature_values) == expected_length
    assert feature_values == sorted(feature_values)


def test_interpolate_feature_prob():
    """
    Test that the feature probability interpolation is a float and has a coherent value.
    """

    distribution = [9794, 12961, 14914, 16865, 18687, 20763, 23357, 27069, 33514, 50271]
    feature_value = 16000
    interpolation = interpolate_feature_prob(feature_value, distribution)

    assert isinstance(interpolation, float)
    assert 0.3 <= interpolation <= 0.4


def test_compute_distribution():
    """
    Test that computed distribution is exactly the same as expected
    """
    expected = [
        {"feature": 10330.0, "decile": "D1"},
        {"feature": 12958.7, "decile": "D2"},
        {"feature": 14799.0, "decile": "D3"},
        {"feature": 16629.4, "decile": "D4"},
        {"feature": 18402.3, "decile": "D5"},
        {"feature": 21097.6, "decile": "D6"},
        {"feature": 24057.4, "decile": "D7"},
        {"feature": 28264.0, "decile": "D8"},
        {"feature": 34245.5, "decile": "D9"},
    ]

    df = pd.read_csv("tests/nantes_enriched.csv")

    assert compute_distribution(df).round(1).to_dict(orient="records") == expected
