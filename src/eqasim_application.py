"""
Application based on eqasim pipeline
"""
# %%
# Init
import pandas as pd
from tools import add_attributes_households, read_filosofi
from functions import (
    compute_crossed_probabilities,
    compute_distribution,
    compute_vec_all,
    compute_p_r,
    run_assignment,
)

pd.set_option("mode.chained_assignment", None)


# %%
# Parameters
MODALITIES = {
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

CODE_INSEE = "44109"

# %%
# Prepare data
df_population = pd.read_csv(
    "../data/inputs/eqasim_population_44.csv", sep=";", dtype={"commune_id": str}
)
df_households = pd.read_csv(
    "../data/inputs/eqasim_households_44.csv", sep=";", dtype={"commune_id": str}
)
df_households = add_attributes_households(df_population, df_households)
df_income_imputed = pd.read_csv(
    "../data/inputs/eqasim_imputed_income_44.csv", sep=";", dtype={"commune_id": str}
)
df_income_attributes = read_filosofi(
    "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
)

# %%
# RUN
synth_pop = df_households.query(f"commune_id == '{CODE_INSEE}'")
crossed_probabilities = compute_crossed_probabilities(synth_pop, MODALITIES)
filosofi = compute_distribution(df_income_attributes, df_income_imputed, CODE_INSEE, MODALITIES)
vec_all_incomes = compute_vec_all(filosofi)
p_R = compute_p_r(vec_all_incomes, df_income_imputed, CODE_INSEE)

res = run_assignment(filosofi, vec_all_incomes, crossed_probabilities, MODALITIES)

# %%
