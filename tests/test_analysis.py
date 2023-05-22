from bhepop2.analysis import *
import pandas as pd


def test_analyse_enriched_population(filosofi_distributions_nantes, test_modalities, tmp_dir):
    pop = pd.read_csv("tests/nantes_enriched.csv")
    pop2 = pd.read_csv("tests/nantes_enriched_gradient2.csv")

    populations = {"base": pop, "gradient": pop2}

    analyse_enriched_populations(
        populations, filosofi_distributions_nantes, "Filosofi", test_modalities, tmp_dir
    )


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
