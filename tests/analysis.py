"""
Result analysis
"""
import os
from bhepop2.max_entropy_enrich import MaxEntropyEnrichment
from tests.conftest import *
import numpy as np
import plotly.express as px


PLOTDIR = "tests/plots/"


def compute_distribution(df):
    """ """
    return pd.DataFrame(
        {
            "feature": np.percentile(
                df["feature"],
                np.arange(0, 100, 10),
            )[1:],
            "decile": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
        }
    )


def plot_analysis(attribute, modality):
    """ """
    fig = px.bar(
        df_analysis[
            (df_analysis["attribute"] == attribute) & (df_analysis["modality"] == modality)
        ],
        x="decile",
        y="feature",
        color="source",
        barmode="group",
        title=f"{attribute} - {modality}",
    )
    fig.write_image(f"{PLOTDIR}/{attribute}_{modality}.png", format="png", width=1000, height=600)


if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


# read data
expected_enriched_pop = pd.read_csv("tests/nantes_enriched.csv")
filosofi = get_filosofi_distributions()

# filter and format filosofi
filosofi = filosofi.query(f"commune_id=='{CODE_INSEE}'")
del filosofi["commune_id"]
del filosofi["reference_median"]

filosofi = filosofi.melt(
    id_vars=["attribute", "modality"],
    value_vars=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    value_name="feature",
    var_name="decile",
)
filosofi["source"] = "Filosofi"

# compute dataframe of distribution from enriched population

df_analysis = compute_distribution(expected_enriched_pop)
df_analysis["attribute"] = "all"
df_analysis["modality"] = "all"

for attribute in MODALITIES.keys():
    for modality in MODALITIES[attribute]:
        distribution = compute_distribution(
            expected_enriched_pop[expected_enriched_pop[attribute] == modality]
        )
        distribution["attribute"] = attribute
        distribution["modality"] = modality

        df_analysis = pd.concat([df_analysis, distribution])

df_analysis["source"] = "bhepop2"

# add filosofi data
df_analysis = pd.concat([df_analysis, filosofi])

# plots
plot_analysis("all", "all")
for attribute in MODALITIES.keys():
    for modality in MODALITIES[attribute]:
        plot_analysis(attribute, modality)
