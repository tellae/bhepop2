"""""" ""


# %%
# init

from tqdm import tqdm
import utils
import pandas as pd
import functions
from tools import read_filosofi

# Set display options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)
pd.set_option("mode.chained_assignment", None)

PATH_PROCESSED = "../data/processed/"
PATH_INPUTS = "../data/inputs/"
PATH_RAW = "../data/raw/"
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

# %%
# script 1 : read synthetic population
# The dataframe synth_pop is the synthetic household population for the city of nantes.
# We have 157,647 households. Each row of synth_pop is therefore a household.

synth_pop = pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")
crossed_probabilities = functions.compute_crossed_probabilities(synth_pop, MODALITIES)

# %%
# script 2 : read data incomes distribution from xls INSEE raw data
df_income_attributes = read_filosofi(
    "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
).query(f"commune_id == '{CODE_INSEE}'")

filosofi = df_income_attributes.copy()
filosofi = filosofi[filosofi["attribute"].isin(functions.get_attributes(MODALITIES))]
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

vec_all_incomes = functions.compute_vec_all(filosofi)
p_R = functions.compute_p_r(vec_all_incomes, df_income_attributes, CODE_INSEE)

# %%
# not used

# all combinations of modalities
all_combinations = crossed_probabilities[functions.get_attributes(MODALITIES)]
all_combinations["total"] = all_combinations.apply(
    lambda x: x["ownership"] + "_" + x["age"] + "_" + x["size"] + "_" + x["family_comp"],
    axis=1,
)

tmp = pd.melt(
    all_combinations,
    id_vars="total",
    value_vars=["ownership", "age", "size", "family_comp"],
)
tmp["key"] = 1
# matrice des moments
tmp = tmp.pivot(index=["variable", "value"], columns="total", values="key")

# TODO add constant
# TODO remove one last modality per variable in line
# TODO correctly sort lines and columns
# TODO replace NaN by 0 and 1.0 by 1
# TODO add validation with R script

# %%
# run assignment
res = functions.run_assignment(filosofi, vec_all_incomes, crossed_probabilities, MODALITIES)

# test : pas de prob négative et les sommes valent 1

# clean income probs

# Ce qu'on a comme probas ici :
# matrice M(i,j) pour i dans les modalités croisées (360)
# et j dans les incomes (190)
# M(i,j) est la probabilité d'être dans la modalité croisée i
# sachant qu'on a un income entre Ij et Ij+1
# On veut l'inverse : la proba d'avoir l'income [Ij, Ij+1] sachant
# la modalité croisée

# donc on veut inverser la proba (Bayes). On connait la proba de chaque modalité croisée
# (fréquence dans la pop. synthétique). Potentiellement 0, dans ce cas
# on ne calcule pas la proba car la modalité croisée n'est pas présente.

# TODO : vérifier ordre bien conservé
# print(res)
# print(len(res))
# print(crossed_probabilities)

for i in range(3):
    res[i] = res[i] * p_R["proba1"][i]
res["sum"] = res.sum(axis=1)
for i in range(3):
    res[i] = res[i] / res["sum"]
res["sum"] = res.sum(axis=1)


# et là normalement c'est fini

# si négatif, élargir les intervalles de revenus

# TODO : pour finir, tirage des revenus :
# tirer autant d'intervalles que d'individus pour une modalité croisée
# donnée (et une fois l'intervalle tiré, tirer une valeur dans l'intervalle)
# on termine avec un vecteur d'incomes pour les individus de ce type.

# TODO : prendre la population synthétique synth_pop et tirer une tranche de revenu, puis un revenu
# %%

# # New version of read filosofi
# Running model for income 0
# SUCCESS on income 0 with fun=0.0006522246419895517
# Running model for income 1
# SUCCESS on income 1 with fun=0.0006522246419894961
# Running model for income 2
# SUCCESS on income 2 with fun=0.000738048106135869


# # Old version
# Running model for income 0
# SUCCESS on income 0 with fun=0.0006522246419895517
# Running model for income 1
# SUCCESS on income 1 with fun=0.0006522246419894961
# Running model for income 2
# SUCCESS on income 2 with fun=0.000738048106135869

res_old = pd.read_pickle("../data/output/res_old.pkl")

# %%
