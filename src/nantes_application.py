# %% init

import pandas as pd

DECILE_0 = 0
DECILE_10 = 1.5

PATH_PROCESSED = "../data/processed/"
ORIGINAL_SYNTHETIC_POPULATION = "synth_pop_original.feather"
FILOSI_DECILES = "deciles_filosofi.feather"

# %% script 1

# read raw synthetic population
# TODO read raw data and expand with IPONDI, keep attributes for Nantes city
synth_pop = pd.read_feather(PATH_PROCESSED + ORIGINAL_SYNTHETIC_POPULATION)
print(synth_pop.head())
print(synth_pop.describe())

# The dataframe synth_pop is the synthetic household population for the city of nantes.
# We have 157,647 households. Each row of synth_pop is therefore a household.

synth_pop["key"] = 1
group = synth_pop.groupby(["age", "size", "ownership", "family_comp"],
                          as_index=False)["key"].sum()
group = group.sort_values(by=["family_comp", "size", "age", "ownership"],
                          ascending=[False, True, True, True])
group["probability"] = group["key"] / sum(group["key"])
group = group[["ownership", "age", "size", "family_comp", "probability"]]
# TODO need to sort correctly the final table

# %% script 2
decile_total = pd.read_feather(PATH_PROCESSED + FILOSI_DECILES)
decile_total["D0"] = DECILE_0
decile_total["D10"] = decile_total["D9"] * DECILE_10
decile_total = decile_total[
    ["modality"] + list(map(lambda a: "D" + str(a), list(range(0, 11))))]

# get all deciles and sort values
vec_all_incomes = []
for index, row in decile_total.iterrows():
    for r in list(map(lambda a: "D" + str(a), list(range(1, 11)))):
        vec_all_incomes.append(row[r])
vec_all_incomes.sort()  # 190 modalities for the income

# TODO get those data from filosofi files
total_population_decile = [
    10303.48,  # values from raw filosofi files
    13336.07,
    16023.85,
    18631.33,
    21262.67,
    24188.00,
    27774.44,
    32620.00,
    41308.00,
    75090.00  # max des déciles
]


# %% linear extrapolation of these 190 incomes from the total population deciles
# TODO move function in an utils.py import
# TODO document function params
def interpolate_income(income, distribution):
    distribution = [0] + distribution
    decile_top = 0
    while income > distribution[decile_top]:
        decile_top += 1

    interpolation = (income - distribution[decile_top - 1]) * (
        decile_top * 0.1 -
        (decile_top - 1) * 0.1) / (distribution[decile_top] - distribution[
            decile_top - 1]) + (decile_top - 1) * 0.1

    return interpolation


p_R = pd.DataFrame({"income": vec_all_incomes})
p_R["proba1"] = p_R.apply(
    lambda x: interpolate_income(x["income"], total_population_decile), axis=1)

# %%
