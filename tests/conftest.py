from bhepop2.tools import read_filosofi, filosofi_attributes
from bhepop2.sources.marginal_distributions import QuantitativeMarginalDistributions
from bhepop2.functions import get_attributes

import pytest
import pandas as pd
import os
import shutil

SEED = 42

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
    "delta_min": 1000,
}

TEST_DATA_FOLDER = "tests/data/"
TMP_DIR = "tests/tmp/"


@pytest.fixture(scope="session")
def tmp_dir():
    return TMP_DIR


@pytest.fixture(scope="session", autouse=True)
def tmp_dir_create_delete(tmp_dir):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    yield

    shutil.rmtree(tmp_dir)


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
def test_feature_name():
    return "test_feature"


@pytest.fixture(scope="session")
def test_seed():
    return SEED


@pytest.fixture(scope="session")
def filosofi_distributions_nantes(test_insee_code):
    filosofi = read_filosofi(
        TEST_DATA_FOLDER + "FILO_DISP_COM.xls", "15", filosofi_attributes, [test_insee_code]
    )

    return filosofi


@pytest.fixture(scope="session")
def filosofi_global_distribution_nantes(filosofi_distributions_nantes):
    return filosofi_distributions_nantes[filosofi_distributions_nantes["modality"] == "all"]


@pytest.fixture(scope="function")
def quantitative_marginal_distributions(
    filosofi_distributions_nantes, test_modalities, test_parameters
):
    return QuantitativeMarginalDistributions(
        filosofi_distributions_nantes,
        attribute_selection=list(test_modalities.keys()),
        **test_parameters
    )


@pytest.fixture(scope="session")
def synthetic_population_nantes():
    return pd.read_csv(TEST_DATA_FOLDER + "synpop_nantes.csv")


@pytest.fixture(scope="session")
def pop_synt_men_nantes():
    synt_pop = pd.read_csv("tests/data/pop_synth_men_nantes.csv")
    return synt_pop


@pytest.fixture(scope="session")
def expected_enriched_population_nantes():
    return pd.read_csv(TEST_DATA_FOLDER + "nantes_enriched.csv")


@pytest.fixture(scope="session")
def eqasim_population():
    return pd.read_csv(TEST_DATA_FOLDER + "eqasim_population_0.001.csv", sep=";")


@pytest.fixture(scope="session")
def eqasim_households():
    return pd.read_csv(TEST_DATA_FOLDER + "eqasim_households_0.001.csv", sep=";")
