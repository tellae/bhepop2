# %% init

import pandas as pd

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

# %%

synth_pop["key"] = 1
group = synth_pop.groupby(["age", "size", "ownership", "family_comp"],
                          as_index=False)["key"].sum()
group = group.sort_values(by=["family_comp", "size", "age", "ownership"],
                          ascending=[False, True, True, True])
group["probability"] = group["key"] / sum(group["key"])
group = group[["ownership", "age", "size", "family_comp", "probability"]]
# TODO need to sort correctly the final table
# %%
