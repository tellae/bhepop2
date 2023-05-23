from bhepop2.analysis import *


def test_analyse_enriched_population(
    filosofi_distributions_nantes, expected_enriched_population_nantes, test_modalities, tmp_dir
):
    populations = {"enriched": expected_enriched_population_nantes}

    analyse_enriched_populations(
        populations,
        filosofi_distributions_nantes,
        "Filosofi",
        test_modalities,
        tmp_dir,
        plots=False,
    )


def test_compute_distribution(expected_enriched_population_nantes):
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

    assert (
        compute_distribution(expected_enriched_population_nantes).round(1).to_dict(orient="records")
        == expected
    )
