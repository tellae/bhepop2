"""""" ""


# %%
# init

import pandas as pd

DECILE_0 = 0
DECILE_10 = 1.5

PATH_PROCESSED = "../data/processed/"
PATH_INPUTS = "../data/inputs/"
PATH_RAW = "../data/raw/"
SYNTHETIC_POP = "nantes_synth_pop.csv"
FILOSI_DECILES = "deciles_filosofi.feather"
FILOSOFI = "indic-struct-distrib-revenu-2015-COMMUNES/"
CODE_INSEE = "44109"

FILOSOFI_MODALITIES = [
    {"name": "1 person", "sheet": "TAILLEM_1", "col_pattern": "TME1"},
    {"name": "2 persons", "sheet": "TAILLEM_2", "col_pattern": "TME2"},
    {"name": "3 persons", "sheet": "TAILLEM_3", "col_pattern": "TME3"},
    {"name": "4 persons", "sheet": "TAILLEM_4", "col_pattern": "TME4"},
    {"name": "5 persons or more", "sheet": "TAILLEM_5", "col_pattern": "TME5"},
    {"name": "Single man", "sheet": "TYPMENR_1", "col_pattern": "TYM1"},
    {"name": "Single woman", "sheet": "TYPMENR_2", "col_pattern": "TYM2"},
    {"name": "Couple without children", "sheet": "TYPMENR_3", "col_pattern": "TYM3"},
    {"name": "Couple with children", "sheet": "TYPMENR_4", "col_pattern": "TYM4"},
    {"name": "Single parent family", "sheet": "TYPMENR_5", "col_pattern": "TYM5"},
    {"name": "Complex households", "sheet": "TYPMENR_6", "col_pattern": "TYM6"},
    {"name": "0_29", "sheet": "TRAGERF_1", "col_pattern": "AGE1"},
    {"name": "30_39", "sheet": "TRAGERF_2", "col_pattern": "AGE2"},
    {"name": "40_49", "sheet": "TRAGERF_3", "col_pattern": "AGE3"},
    {"name": "50_59", "sheet": "TRAGERF_4", "col_pattern": "AGE4"},
    {"name": "60_74", "sheet": "TRAGERF_5", "col_pattern": "AGE5"},
    {"name": "75_or_more", "sheet": "TRAGERF_6", "col_pattern": "AGE6"},
    {"name": "Owner", "sheet": "OCCTYPR_1", "col_pattern": "TOL1"},
    {"name": "Tenant", "sheet": "OCCTYPR_2", "col_pattern": "TOL2"},
]

# Set display options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)
pd.set_option("mode.chained_assignment", None)

# %%
# script 1
##################################################################

# read raw synthetic population
# TODO read raw data and expand with IPONDI, keep attributes for Nantes city
# TODO check if synthetic pop is related to population or households
synth_pop = pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")

# The dataframe synth_pop is the synthetic household population for the city of nantes.
# We have 157,647 households. Each row of synth_pop is therefore a household.

synth_pop["key"] = 1
group = synth_pop.groupby(["age", "size", "ownership", "family_comp"], as_index=False)[
    "key"
].sum()
group = group.sort_values(
    by=["family_comp", "size", "age", "ownership"], ascending=[False, True, True, True]
)
group["probability"] = group["key"] / sum(group["key"])
group = group[["ownership", "age", "size", "family_comp", "probability"]]
# TODO need to sort correctly the final table

# TODO add validation with R script

# script 2
# %%
# script 2 : read data revenu distribution from xls INSEE raw data
filosofi = pd.DataFrame()

for modality in FILOSOFI_MODALITIES:
    SHEET = modality["sheet"]
    COL_PATTERN = modality["col_pattern"]
    tmp = pd.read_excel(
        PATH_RAW + FILOSOFI + "FILO_DISP_COM.xls",
        sheet_name=SHEET,
        skiprows=5,
    ).query("CODGEO=='" + CODE_INSEE + "'")
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
decile_total = decile_total[
    ["modality"] + list(map(lambda a: "D" + str(a), list(range(0, 11))))
]

# get all deciles and sort values
vec_all_incomes = []
for index, row in decile_total.iterrows():
    for r in list(map(lambda a: "D" + str(a), list(range(1, 11)))):
        vec_all_incomes.append(row[r])
vec_all_incomes.sort()  # 190 modalities for the income

# %%
# script 2 : get total population revenu distribution
filosofi_all = pd.read_excel(
    PATH_RAW + FILOSOFI + "FILO_DISP_COM.xls",
    sheet_name="ENSEMBLE",
    skiprows=5,
).query("CODGEO=='" + CODE_INSEE + "'")


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
# TODO move function in an utils.py import
# TODO document function params
def interpolate_income(income, distribution):
    distribution = [0] + distribution
    decile_top = 0
    while income > distribution[decile_top]:
        decile_top += 1

    interpolation = (income - distribution[decile_top - 1]) * (
        decile_top * 0.1 - (decile_top - 1) * 0.1
    ) / (distribution[decile_top] - distribution[decile_top - 1]) + (
        decile_top - 1
    ) * 0.1

    return interpolation


p_R = pd.DataFrame({"income": vec_all_incomes})
p_R["proba1"] = p_R.apply(
    lambda x: interpolate_income(x["income"], total_population_decile), axis=1
)

# TODO add validation with R script

# %%
# script 3
#########################################################

# all combinations of modalities
# TODO order correctly modalities
all_combinations = group[["ownership", "age", "size", "family_comp"]]
all_combinations["total"] = all_combinations.apply(
    lambda x: x["ownership"]
    + "_"
    + x["age"]
    + "_"
    + x["size"]
    + "_"
    + x["family_comp"],
    axis=1,
)

tmp = pd.melt(
    all_combinations,
    id_vars="total",
    value_vars=["ownership", "age", "size", "family_comp"],
)
tmp["key"] = 1
tmp = tmp.pivot(index=["variable", "value"], columns="total", values="key")
print(tmp.head())

# TODO add constant
# TODO remove one last modality per variable in line
# TODO correctly sort lines and columns
# TODO replace NaN by 0 and 1.0 by 1
# TODO add validation with R script

# %%
