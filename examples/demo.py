#!/usr/bin/env python
# coding: utf-8

# # Adding income information to a synthetic population of households
# 
# In this example, we already have a synthetic population of households on Nantes city. This synthetic population was built using the French national Census of the population. For each household, serveral characteristics have been added:
# 
# - Ownership : owner or tenant of its accomodation
# - Age : age of reference person
# - Size : number of persons
# - Family composition : composition (single person, couple with ou without children, etc)
# 
# The objectif is to add income information to each household. In order to reach this goal, we use another data source named Filosofi. More precisely, this data source gives information on the income distribution (deciles) for each city, per household characteristics.
# 
# Filosofi is an indicator set implemented by INSEE which is the French National Institute of Statistics. See [insee.fr](https://www.insee.fr/fr/metadonnees/source/serie/s1172) for more details.
# 

# In[1]:

import pdb # pov debug

import warnings
import pandas as pd
from bhepop2.gradient_enrich import MaxEntropyEnrichment_gradient
from bhepop2.tools import read_filosofi, compute_distribution, plot_analysis

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("mode.chained_assignment", None)

# ## Data preparation
# 
# Read synthetic population which doesn't contain revenu information.

# In[2]:


synth_pop = pd.read_csv("../data/inputs/nantes_synth_pop.csv", sep=";")

synth_pop.head()


# Read Filosofi data and format dataframe.

# In[3]:


filosofi = read_filosofi(
    "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
).query(f"commune_id == '44109'")

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

filosofi.head()


# ## Run algorithm

# In[4]:


# Household modalities
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

# Algorithm parameters
PARAMETERS = {
    "abs_minimum": 0,
    "relative_maximum": 1.2,
    "maxentropy_algorithm": "Nelder-Mead",
    "maxentropy_verbose": 0,
    "delta_min": 500,
}

# Optimisation preparation
enrich_class = MaxEntropyEnrichment_gradient(
    synth_pop, filosofi, list(MODALITIES.keys()), parameters=PARAMETERS, seed=42
)

# Run optimisation
enrich_class.optimise()

# Assign data to synthetic population
pop = enrich_class.assign_feature_value_to_pop()

print(pop.head())


# ## Results analysis

# ### Data preparation

# Format Filosofi data for comparison.

# In[5]:


filosofi_formated = filosofi.copy()
del filosofi_formated["commune_id"]
del filosofi_formated["reference_median"]

filosofi_formated = filosofi_formated.melt(
    id_vars=["attribute", "modality"],
    value_vars=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    value_name="feature",
    var_name="decile",
)
filosofi_formated["source"] = "Filosofi"


# Format simulation data for comparison.

# In[6]:


# distribution of all households
df_analysis = compute_distribution(pop)
df_analysis["attribute"] = "all"
df_analysis["modality"] = "all"

# distribution of each modality
for attribute in MODALITIES.keys():
    for modality in MODALITIES[attribute]:
        distribution = compute_distribution(pop[pop[attribute] == modality])
        distribution["attribute"] = attribute
        distribution["modality"] = modality

        df_analysis = pd.concat([df_analysis, distribution])

df_analysis["source"] = "bhepop2"


# Merge observed Filosofi and simulation data.

# In[7]:


# add filosofi data
df_analysis = pd.concat([df_analysis, filosofi_formated])

# format data
df_analysis = df_analysis.pivot(
    columns="source", index=["attribute", "modality", "decile"]
).reset_index()
df_analysis.columns = ["attribute", "modality", "decile", "Filosofi", "bhepop2"]


# ### Some plots

# In[8]:


from IPython.display import Image


# In[9]:


Image(plot_analysis(df_analysis, "all", "all").to_image())


# In[10]:


Image(plot_analysis(df_analysis, "ownership", "Tenant").to_image())


# In[11]:


Image(plot_analysis(df_analysis, "age", "30_39").to_image())


# In[12]:


Image(plot_analysis(df_analysis, "size", "3_pers").to_image())


# In[13]:


Image(plot_analysis(df_analysis, "family_comp", "Single_parent").to_image())


# %%
