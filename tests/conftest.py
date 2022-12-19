from src.tools import read_filosofi

import pandas as pd

PATH_INPUTS = "../data/inputs/"
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
    "delta_min": 1000
}

def get_synth_pop_nantes():
    synth_pop = pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")

    return synth_pop

def get_filosofi_distributions():
    df_income_attributes = read_filosofi(
        "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
    )
    filosofi = df_income_attributes.copy()
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

    return filosofi