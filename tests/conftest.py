from bhepop2.tools import read_filosofi
from bhepop2.functions import get_attributes

import pytest
import pandas as pd

SEED = 42

PATH_INPUTS = "data/inputs/"
SYNTHETIC_POP = "nantes_synth_pop.csv"
CODE_INSEE = "44109"
MODALITIES = {
    "ownership": ["Owner", "Tenant"],
    "age": ["0_29", "30_39", "40_49", "50_59", "60_74", "75_or_more"],
    "size": ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"],
    "family_comp": [
        "Single_man",
        "Single_wom",
        "Couple_without_child",
        "Couple_with_child",
        "Single_parent",
        "complex_hh",
    ],
}

parameters = {
    "abs_minimum": 0,
    "relative_maximum": 1.5,
    "maxentropy_algorithm": "Nelder-Mead",
    "maxentropy_verbose": 0,
    "delta_min": 1000,
}


@pytest.fixture(scope="session")
def test_insee_code():
    return CODE_INSEE


@pytest.fixture(scope="session")
def test_modalities():
    return MODALITIES


@pytest.fixture(scope="session")
def test_attributes(test_modalities):
    return get_attributes(test_modalities)


@pytest.fixture(scope="session")
def test_parameters():
    return parameters


@pytest.fixture(scope="session")
def test_seed():
    return SEED


@pytest.fixture(scope="session")
def synthetic_population_nantes():
    return pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")


@pytest.fixture(scope="session")
def filosofi_distributions_nantes(test_insee_code):
    filosofi = read_filosofi("data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls", "15")
    filosofi.rename(
        columns={
            "q1": "D1",
            "q2": "D2",
            "q3": "D3",
            "q4": "D4",
            "q5": "D5",
            "q6": "D6",
            "q7": "D7",
            "q8": "D8",
            "q9": "D9",
        },
        inplace=True,
    )
    filosofi = filosofi.query(f"commune_id == '{test_insee_code}'")

    return filosofi

@pytest.fixture(scope="session")
def eqasim_population():
    return pd.read_csv(PATH_INPUTS + "eqasim_population_0.001.csv", sep=";")

@pytest.fixture(scope="session")
def eqasim_households():
    return pd.read_csv(PATH_INPUTS + "eqasim_households_0.001.csv", sep=";")
