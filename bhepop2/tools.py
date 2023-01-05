"""
Utility functions
"""

import pandas as pd


def read_filosofi(path2file: str) -> pd.DataFrame:
    """
    Read Filosofi data
    :param path2file: path to raw xlsx file

    :return dict of dataframe per attribute
    """

    FILOSOFI_ATTRIBUTES = [
        {
            "name": "all",
            "modalities": [
                {"name": "all", "sheet": "ENSEMBLE", "col_pattern": ""},
            ],
        },
        {
            "name": "size",
            "modalities": [
                {"name": "1_pers", "sheet": "TAILLEM_1", "col_pattern": "TME1"},
                {"name": "2_pers", "sheet": "TAILLEM_2", "col_pattern": "TME2"},
                {"name": "3_pers", "sheet": "TAILLEM_3", "col_pattern": "TME3"},
                {"name": "4_pers", "sheet": "TAILLEM_4", "col_pattern": "TME4"},
                {"name": "5_pers_or_more", "sheet": "TAILLEM_5", "col_pattern": "TME5"},
            ],
        },
        {
            "name": "family_comp",
            "modalities": [
                {"name": "Single_man", "sheet": "TYPMENR_1", "col_pattern": "TYM1"},
                {"name": "Single_wom", "sheet": "TYPMENR_2", "col_pattern": "TYM2"},
                {"name": "Couple_without_child", "sheet": "TYPMENR_3", "col_pattern": "TYM3"},
                {"name": "Couple_with_child", "sheet": "TYPMENR_4", "col_pattern": "TYM4"},
                {"name": "Single_parent", "sheet": "TYPMENR_5", "col_pattern": "TYM5"},
                {"name": "complex_hh", "sheet": "TYPMENR_6", "col_pattern": "TYM6"},
            ],
        },
        {
            "name": "age",
            "modalities": [
                {"name": "0_29", "sheet": "TRAGERF_1", "col_pattern": "AGE1"},
                {"name": "30_39", "sheet": "TRAGERF_2", "col_pattern": "AGE2"},
                {"name": "40_49", "sheet": "TRAGERF_3", "col_pattern": "AGE3"},
                {"name": "50_59", "sheet": "TRAGERF_4", "col_pattern": "AGE4"},
                {"name": "60_74", "sheet": "TRAGERF_5", "col_pattern": "AGE5"},
                {"name": "75_or_more", "sheet": "TRAGERF_6", "col_pattern": "AGE6"},
            ],
        },
        {
            "name": "ownership",
            "modalities": [
                {"name": "Owner", "sheet": "OCCTYPR_1", "col_pattern": "TOL1"},
                {"name": "Tenant", "sheet": "OCCTYPR_2", "col_pattern": "TOL2"},
            ],
        },
        {
            "name": "income_source",
            "modalities": [
                {"name": "Salary", "sheet": "OPRDEC_1", "col_pattern": "OPR1"},
                {"name": "Unemployment", "sheet": "OPRDEC_2", "col_pattern": "OPR2"},
                {"name": "Independent", "sheet": "OPRDEC_3", "col_pattern": "OPR3"},
                {"name": "Pension", "sheet": "OPRDEC_4", "col_pattern": "OPR4"},
                {"name": "Property", "sheet": "OPRDEC_5", "col_pattern": "OPR5"},
                {"name": "None", "sheet": "OPRDEC_6", "col_pattern": "OPR6"},
            ],
        },
    ]

    # build full list of sheets
    sheet_list = []
    for attribute in FILOSOFI_ATTRIBUTES:
        sheet_list = sheet_list + [x["sheet"] for x in attribute["modalities"]]

    # read all needed sheets
    excel_df = pd.read_excel(
        path2file,
        sheet_name=sheet_list,
        skiprows=5,
    )

    df = pd.DataFrame()
    for attribute in FILOSOFI_ATTRIBUTES:
        for modality in attribute["modalities"]:
            sheet = modality["sheet"]
            col_pattern = modality["col_pattern"]

            # Load income distribution
            data = excel_df[sheet][
                ["CODGEO"]
                + [
                    "%sD%d15" % (col_pattern, q) if q != 5 else col_pattern + "Q215"
                    for q in range(1, 10)
                ]
            ]
            data.columns = ["commune_id", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]
            data["reference_median"] = data["q5"]
            data["modality"] = modality["name"]
            data["attribute"] = attribute["name"]
            df = pd.concat([df, data])

    # Validation
    assert len(FILOSOFI_ATTRIBUTES) == len(df["attribute"].unique())
    assert len(sheet_list) == len(df["modality"].unique())

    return df[
        [
            "commune_id",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "reference_median",
            "attribute",
            "modality",
        ]
    ]


def add_attributes_households(population: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    """
    Add attributes to households
    - Households size (number of persons)
    - Households type (composition of persons)

    :param population: dataframe of population
    :param households: dataframe of households

    return: households dataframe with new attributes
    """

    df_population = population.copy()
    df_households = households.copy()

    # Add household size
    df_households_size = df_population.groupby(["household_id"], as_index=False)[
        "person_id"
    ].count()
    df_households_size.rename(columns={"person_id": "size"}, inplace=True)
    df_households_size["size"] = df_households_size["size"].map(
        lambda x: str(x) + "_pers" if x < 5 else "5_pers_or_more"
    )
    df_households = pd.merge(df_households, df_households_size)

    # Add household type
    # compute single man or woman
    # household of 1 person
    df_households_info = df_population.groupby(["household_id"], as_index=False).agg(
        {"household_size": "first", "sex": "first"}
    )
    df_households_single_man = df_households_info.query(
        "household_size==1 and sex=='male'"
    ).reset_index()
    df_households_single_man["family_comp"] = "Single_man"
    df_households_single_woman = df_households_info.query(
        "household_size==1 and sex=='female'"
    ).reset_index()
    df_households_single_woman["family_comp"] = "Single_wom"

    # compute couple without child
    # 2 persons and all have couple=True
    df_households_couple = (
        df_population.query("couple==True")
        .groupby(["household_id"], as_index=False)
        .agg({"couple": "count", "household_size": "first"})
        .query("couple==2")
    )
    df_households_couple_without_child = df_households_couple.query(
        "couple==2 and household_size==2"
    ).reset_index()
    df_households_couple_without_child["family_comp"] = "Couple_without_child"

    # compute couple with child
    # 3 persons or more, 2 persons in couple, and max age of person non couple < 25
    df_households_child = (
        df_population.query("age<25 and couple==False")
        .groupby(["household_id"], as_index=False)
        .agg({"person_id": "count"})
        .rename(columns={"person_id": "child_count"})
    )
    df_households_couple_with_child = (
        df_households_couple.merge(df_households_child, how="left", on="household_id")
        .query("household_size==child_count+2")
        .reset_index()
    )
    df_households_couple_with_child["family_comp"] = "Couple_with_child"
    # compute single parent
    # 2 persons or more, no one in couple, oldest person = [25-60[, and others age < 25
    df_households_no_couple = df_population.query(
        "couple==False and household_size>=2"
    ).sort_values(["household_id", "age"], ascending=[True, False])

    df_households_no_couple_oldest = (
        df_households_no_couple.groupby(["household_id"], as_index=False)
        .agg({"age": "first", "person_id": "first"})
        .rename(columns={"age": "oldest_age", "person_id": "oldest_person_id"})
    )

    df_households_no_couple_second_oldest = (
        df_households_no_couple.merge(df_households_no_couple_oldest, how="left")
        .query("person_id!=oldest_person_id")
        .sort_values(["household_id", "age"], ascending=[True, False])
        .groupby(["household_id"], as_index=False)
        .agg({"age": "first"})
        .rename(columns={"age": "second_oldest_age"})
    )

    df_households_single_parent = df_households_no_couple_oldest.merge(
        df_households_no_couple_second_oldest, how="left", on="household_id"
    )
    df_households_single_parent = df_households_single_parent.query(
        "oldest_age>=25 and oldest_age<60 and second_oldest_age<25"
    ).reset_index()
    df_households_single_parent["family_comp"] = "Single_parent"

    # combine type
    df_households = (
        df_households.merge(
            df_households_single_man[["household_id", "family_comp"]],
            how="left",
            on="household_id",
            suffixes=("", "_1"),
        )
        .merge(
            df_households_single_woman[["household_id", "family_comp"]],
            how="left",
            on="household_id",
            suffixes=("_1", "_2"),
        )
        .merge(
            df_households_couple_without_child[["household_id", "family_comp"]],
            how="left",
            on="household_id",
            suffixes=("_2", "_3"),
        )
        .merge(
            df_households_couple_with_child[["household_id", "family_comp"]],
            how="left",
            on="household_id",
            suffixes=("_3", "_4"),
        )
        .merge(
            df_households_single_parent[["household_id", "family_comp"]],
            how="left",
            on="household_id",
            suffixes=("_5", "_5"),
        )
    )
    df_households.columns = [
        "household_id",
        "consumption_units",
        "commune_id",
        "size",
        "family_comp_1",
        "family_comp_2",
        "family_comp_3",
        "family_comp_4",
        "family_comp_5",
    ]
    df_households[
        [
            "family_comp_1",
            "family_comp_2",
            "family_comp_3",
            "family_comp_4",
            "family_comp_5",
        ]
    ] = df_households[
        [
            "family_comp_1",
            "family_comp_2",
            "family_comp_3",
            "family_comp_4",
            "family_comp_5",
        ]
    ].fillna(
        ""
    )
    df_households["family_comp"] = df_households.apply(
        lambda x: str(x["family_comp_1"])
        + str(x["family_comp_2"])
        + str(x["family_comp_3"])
        + str(x["family_comp_4"])
        + str(x["family_comp_5"]),
        axis=1,
    )
    # compute complex households as other households
    df_households["family_comp"] = df_households["family_comp"].map(
        lambda x: "complex_hh" if x == "" else x
    )

    df_households = df_households[
        ["household_id", "consumption_units", "commune_id", "size", "family_comp"]
    ]

    return df_households
