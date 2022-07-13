"""""" ""


# %%
# init

import math
from maxentropy import MinDivergenceModel
from itertools import product
from tqdm import tqdm
import utils as utils
import pandas as pd
import numpy as np

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

synth_pop["key"] = 1
group = synth_pop.groupby(["age", "size", "ownership", "family_comp"], as_index=False)["key"].sum()
group = group.sort_values(
    by=["family_comp", "size", "age", "ownership"], ascending=[False, True, True, True]
)
group["probability"] = group["key"] / sum(group["key"])
group = group[["ownership", "age", "size", "family_comp", "probability"]]

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


# %%
# create dictionary of constraints (element of eta_total in R code)

ech = {}
constraint = {}
for variable in variables:
    print(variable)
    ech[variable] = {}

    decile = filosofi[filosofi["modality"].isin(modalities[variable])]

    for modality in modalities[variable]:
        decile_tmp = decile[decile["modality"].isin([modality])]
        total_population_decile_tmp = [
            float(decile_tmp["D1"]),
            float(decile_tmp["D2"]),
            float(decile_tmp["D3"]),
            float(decile_tmp["D4"]),
            float(decile_tmp["D5"]),
            float(decile_tmp["D6"]),
            float(decile_tmp["D7"]),
            float(decile_tmp["D8"]),
            float(decile_tmp["D9"]),
            float(decile_tmp["D9"]) * DECILE_10,
        ]
        p_R_tmp = pd.DataFrame({"income": vec_all_incomes})
        p_R_tmp["proba1"] = p_R_tmp.apply(
            lambda x: utils.interpolate_income(x["income"], total_population_decile_tmp),
            axis=1,
        )
        ech[variable][modality] = p_R_tmp

    # p
    # get statistics (frequency)
    prob_1 = group.groupby([variable], as_index=False)["probability"].sum()

    # multiply frequencies by each element of ech_compo
    for modality in ech[variable]:
        value = prob_1[prob_1[variable].isin([modality])]
        df = ech[variable][modality]
        df["proba1"] = df["proba1"] * float(
            value["probability"]
        )  # prob(income | modality) * frequency // ech is modified inplace here

    ech_list = []
    for modality in ech[variable]:
        ech_list.append(ech[variable][modality])
    C = pd.concat(
        ech_list,
        axis=1,
    )

    C = C.iloc[:, 1::2]
    C.columns = list(range(0, len(ech[variable])))
    C["Proba"] = C.sum(axis=1)
    p = C[["Proba"]]

    # constraint
    constraint[variable] = {}
    for modality in ech[variable]:
        constraint[variable][modality] = ech[variable][modality]["proba1"] / p["Proba"]

# %%
# optimisation (maxentropy)

samplespace = list(product(ownership, age, size, family_comp))
samplespace = [{variables[i]: x[i] for i in range(len(x))} for x in samplespace]


def f0(x):
    return x in samplespace


def fownership(x):
    return x["ownership"] == "Owner"


def fage_1(x):
    return x["age"] == "0_29"


def fage_2(x):
    return x["age"] == "30_39"


def fage_3(x):
    return x["age"] == "40_49"


def fage_4(x):
    return x["age"] == "50_59"


def fage_5(x):
    return x["age"] == "60_74"


def fsize_1(x):
    return x["size"] == "1_pers"


def fsize_2(x):
    return x["size"] == "2_pers"


def fsize_3(x):
    return x["size"] == "3_pers"


def fsize_4(x):
    return x["size"] == "4_pers"


def fcomp_1(x):
    return x["family_comp"] == "Single_man"


def fcomp_2(x):
    return x["family_comp"] == "Single_wom"


def fcomp_3(x):
    return x["family_comp"] == "Couple_without_child"


def fcomp_4(x):
    return x["family_comp"] == "Couple_with_child"


def fcomp_5(x):
    return x["family_comp"] == "Single_parent"


f = [
    f0,
    fownership,
    fage_1,
    fage_2,
    fage_3,
    fage_4,
    fage_5,
    fsize_1,
    fsize_2,
    fsize_3,
    fsize_4,
    fcomp_1,
    fcomp_2,
    fcomp_3,
    fcomp_4,
    fcomp_5,
]

prior_df = pd.DataFrame.from_dict(samplespace)
prior_df_perc = prior_df.merge(group, how="left", on=variables)
prior_df_perc["probability"] = prior_df_perc.apply(
    lambda x: 0 if x["probability"] != x["probability"] else x["probability"], axis=1
)

prior_df_perc_reducted = prior_df_perc.query("probability > 0")
samplespace_reducted = prior_df_perc_reducted[variables].to_dict(orient="records")


def function_prior_prob(x_array):
    return prior_df_perc_reducted["probability"].apply(math.log)


# %%
# build K


probs = pd.DataFrame()
from scipy.optimize import linprog
from maxentropy.utils import DivergenceError


# loop on incomes
for i in range(100):
    try:
        # we do build the model again because it seemed to break after a failed fit
        model_with_apriori = MinDivergenceModel(
            f,
            samplespace_reducted,
            vectorized=False,
            verbose=False,
            prior_log_pdf=function_prior_prob,
        )
        A = model_with_apriori.F.A

        K = [1]

        for variable in modalities:

            for modality in modalities[variable][:-1]:

                K.append(constraint[variable][modality][i])
        K = np.array(K).reshape(1, len(K))

        I = np.identity(np.shape(K)[1])
        A_eq = np.concatenate([model_with_apriori.F.A, I], axis=1)

        c = np.concatenate([np.zeros(np.shape(A)[1]), np.ones(np.shape(K)[1])], axis=None)

        res = linprog(c, A_eq=A_eq, b_eq=K, method="simplex")

        model_with_apriori.fit(K)

        probs[i] = model_with_apriori.probdist()
        print("SUCCESS on income " + str(i) + " with fun=" + str(res.fun))
    except (Exception, DivergenceError):
        print("ERROR on income " + str(i) + " with fun=" + str(res.fun))

    # model_with_apriori.resetparams()

print(probs)
# %%
