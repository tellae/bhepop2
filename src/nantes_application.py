"""""" ""


# %%
# init

from tqdm import tqdm
import utils
import pandas as pd
import functions

DECILE_0 = 0
DECILE_10 = 1.5

PATH_PROCESSED = "../data/processed/"
PATH_INPUTS = "../data/inputs/"
PATH_RAW = "../data/raw/"
SYNTHETIC_POP = "nantes_synth_pop.csv"
FILOSI_DECILES = "deciles_filosofi.feather"
FILOSOFI = "indic-struct-distrib-revenu-2015-COMMUNES/"
CODE_INSEE = "44109"

# parameters for reading INSEE xlsx Filosofi data
FILOSOFI_MODALITIES = [
    {"name": "1_pers", "sheet": "TAILLEM_1", "col_pattern": "TME1"},
    {"name": "2_pers", "sheet": "TAILLEM_2", "col_pattern": "TME2"},
    {"name": "3_pers", "sheet": "TAILLEM_3", "col_pattern": "TME3"},
    {"name": "4_pers", "sheet": "TAILLEM_4", "col_pattern": "TME4"},
    {"name": "5_pers_or_more", "sheet": "TAILLEM_5", "col_pattern": "TME5"},
    {"name": "Single_man", "sheet": "TYPMENR_1", "col_pattern": "TYM1"},
    {"name": "Single_wom", "sheet": "TYPMENR_2", "col_pattern": "TYM2"},
    {"name": "Couple_without_child", "sheet": "TYPMENR_3", "col_pattern": "TYM3"},
    {"name": "Couple_with_child", "sheet": "TYPMENR_4", "col_pattern": "TYM4"},
    {"name": "Single_parent", "sheet": "TYPMENR_5", "col_pattern": "TYM5"},
    {"name": "complex_hh", "sheet": "TYPMENR_6", "col_pattern": "TYM6"},
    {"name": "0_29", "sheet": "TRAGERF_1", "col_pattern": "AGE1"},
    {"name": "30_39", "sheet": "TRAGERF_2", "col_pattern": "AGE2"},
    {"name": "40_49", "sheet": "TRAGERF_3", "col_pattern": "AGE3"},
    {"name": "50_59", "sheet": "TRAGERF_4", "col_pattern": "AGE4"},
    {"name": "60_74", "sheet": "TRAGERF_5", "col_pattern": "AGE5"},
    {"name": "75_or_more", "sheet": "TRAGERF_6", "col_pattern": "AGE6"},
    {"name": "Owner", "sheet": "OCCTYPR_1", "col_pattern": "TOL1"},
    {"name": "Tenant", "sheet": "OCCTYPR_2", "col_pattern": "TOL2"},
]

# prepare variables and modalities
ownership = ["Owner", "Tenant"]
age = ["0_29", "30_39", "40_49", "50_59", "60_74", "75_or_more"]
size = ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"]
family_comp = [
    "Single_man",
    "Single_wom",
    "Couple_without_child",
    "Couple_with_child",
    "Single_parent",
    "complex_hh",
]
modalities = {
    "ownership": ownership,
    "age": age,
    "size": size,
    "family_comp": family_comp,
}

variables = ["ownership", "age", "size", "family_comp"]

# Set display options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)
pd.set_option("mode.chained_assignment", None)

# %%
# script 1 : read raw insee population

# TODO generate synthetic population from INSEE raw data (see Fabrice R source code)
# raw_insee = pd.read_csv(
#     PATH_RAW + "RP2015_INDCVIZC_txt/FD_INDCVIZC_2015.txt", sep=";", low_memory=False
# )
# raw_insee["COMMUNE"] = raw_insee.apply(lambda x: x["IRIS"][0:5], axis=1)
# raw_insee = raw_insee.query("COMMUNE=='" + CODE_INSEE + "'")
# raw_insee["IPONDI"].sum()

# menages = raw_insee.groupby(["CANTVILLE", "NUMMI"], as_index=False).agg(
#     {"STOCD": ["first"]}
# )
# %%
# script 1 : read synthetic population
##################################################################

# read raw synthetic population
# TODO read raw data and expand with IPONDI, keep attributes for Nantes city
# TODO check if synthetic pop is related to population or households
synth_pop = pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")

# The dataframe synth_pop is the synthetic household population for the city of nantes.
# We have 157,647 households. Each row of synth_pop is therefore a household.
group = functions.group_synthetic_population(synth_pop)

# TODO add validation with R script

# script 2
# %%
# script 2 : read data incomes distribution from xls INSEE raw data
filosofi = pd.DataFrame()

# for modality in FILOSOFI_MODALITIES:
for i in tqdm(range(len(FILOSOFI_MODALITIES))):
    modality = FILOSOFI_MODALITIES[i]
    SHEET = modality["sheet"]
    COL_PATTERN = modality["col_pattern"]
    tmp = utils.read_filosofi(PATH_RAW + FILOSOFI, SHEET, CODE_INSEE)
    tmp["modality"] = modality["name"]
    tmp = tmp.rename(
        columns={
            COL_PATTERN + "D115": "D1",
            COL_PATTERN + "D215": "D2",
            COL_PATTERN + "D315": "D3",
            COL_PATTERN + "D415": "D4",
            COL_PATTERN + "Q215": "D5",
            COL_PATTERN + "D615": "D6",
            COL_PATTERN + "D715": "D7",
            COL_PATTERN + "D815": "D8",
            COL_PATTERN + "D915": "D9",
        }
    )
    tmp = tmp[
        [
            "CODGEO",
            "LIBGEO",
            "modality",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "D7",
            "D8",
            "D9",
        ]
    ]
    filosofi = pd.concat([filosofi, tmp])

decile_total = filosofi.copy()  # pd.read_feather(PATH_PROCESSED + FILOSI_DECILES)
decile_total["D0"] = DECILE_0
decile_total["D10"] = decile_total["D9"] * DECILE_10
decile_total = decile_total[["modality"] + list(map(lambda a: "D" + str(a), list(range(0, 11))))]

# get all deciles and sort values
vec_all_incomes = []
for index, row in decile_total.iterrows():
    for r in list(map(lambda a: "D" + str(a), list(range(1, 11)))):
        vec_all_incomes.append(row[r])
vec_all_incomes.sort()  # 190 modalities for the income

# %%
# script 2 : get total population incomes distribution
# TODO : à quoi sert cette partie ??

filosofi_all = utils.read_filosofi(PATH_RAW + FILOSOFI, "ENSEMBLE", CODE_INSEE)

total_population_decile = [
    filosofi_all["D115"].iloc[0],  # values from raw filosofi files
    filosofi_all["D215"].iloc[0],
    filosofi_all["D315"].iloc[0],
    filosofi_all["D415"].iloc[0],
    filosofi_all["Q215"].iloc[0],
    filosofi_all["D615"].iloc[0],
    filosofi_all["D715"].iloc[0],
    filosofi_all["D815"].iloc[0],
    filosofi_all["D915"].iloc[0],
    max(decile_total["D10"]),  # maximum value of all deciles
]

# %%
# linear extrapolation of these 190 incomes from the total population deciles
p_R = pd.DataFrame({"income": vec_all_incomes})
p_R["proba1"] = p_R.apply(
    lambda x: utils.interpolate_income(x["income"], total_population_decile), axis=1
)

# TODO add validation with R script

# %%
# script 3
#########################################################

# all combinations of modalities
all_combinations = group[["ownership", "age", "size", "family_comp"]]
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

# %% run assignment
import importlib
importlib.reload(functions)
res = functions.run_assignment(filosofi, vec_all_incomes, group, modalities)

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
# print(group)
# for i in res.columns:
#     res.loc[:, i] = res.loc[:, i].div(group.loc[:, "probability"])

# print(res)
# print(len(res))

# TODO : multiplier les incomes (colonnes) par les probas de chaque income
# ces probas sont dans p_R["proba1"]

# tester de nouveau si négatif et somme à 1 
# (somme en ligne, besoin de tous les incomes)

# et là normalement c'est fini

# si négatif, élargir les intervalles de revenus

# pour finir, tirage des revenus :
# tirer autant d'intervalles que d'individus pour une modalité croisée
# donnée (et une fois l'intervalle tiré, tirer une valeur dans l'intervalle)
# on termine avec un vecteur d'incomes pour les individus de ce type.