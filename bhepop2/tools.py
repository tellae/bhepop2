"""
Utility functions
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


## functions for processing synthetic populations


def add_household_size_attribute(
    population: pd.DataFrame,
    values_map: callable = lambda x: str(x) + "_pers" if x < 5 else "5_pers_or_more",
    person_id: str = "person_id",
    household_id: str = "household_id",
    column_name: str = "size",
):
    """
    Add a size attribute to the given synthetic population.

    Even though we add the attribute at a person's level,
    its value is related to the household (we add household size).

    :param population: synthetic population df
    :param values_map: mapping function applied to household count
    :param person_id: name of the column containing persons' ids
    :param household_id: name of the column containing households' ids
    :param column_name: name of the added column

    :return: population with new 'column_name' column containing household size
    """

    # make copy of original population
    population = population.copy()

    # count number of persons by household
    df_households_size = population.groupby([household_id], as_index=False)[person_id].count()

    # rename size column
    df_households_size.rename(columns={person_id: column_name}, inplace=True)

    # map size values if a mapper is provided
    if values_map is not None:
        df_households_size[column_name] = df_households_size[column_name].apply(values_map)

    # add size column to population table
    population = pd.merge(population, df_households_size, on="household_id")

    return population


def add_household_type_attribute(
    population,
    person_id: str = "person_id",
    household_id: str = "household_id",
    column_name="family_comp",
):
    """
    Add a type attribute to the given synthetic population.

    The following attributes are needed on the population:
        - household_size: number of person's in the household
        - sex: person's sex
        - age: person's age
        - couple: boolean indicating if the person is in couple in the household

    The household type can take the following values:
        - Single_man
        - Single_wom
        - Couple_without_child
        - Couple_with_child
        - Single_parent
        - complex_hh

    Even though we add the attribute at a person's level,
    its value is related to the household (we add household type).

    :param population: synthetic population df
    :param person_id: name of the column containing persons' ids
    :param household_id: name of the column containing households' ids
    :param column_name: name of the added column

    :return: population with new 'column_name' column containing household size
    """

    # make copy of original population
    population = population.copy()

    # get single men and women
    df_households_info = population.groupby([household_id], as_index=False).agg(
        {"household_size": "first", "sex": "first"}
    )
    df_households_single_man = df_households_info.query(
        "household_size==1 and sex=='male'"
    ).reset_index()
    df_households_single_man[column_name] = "Single_man"
    df_households_single_woman = df_households_info.query(
        "household_size==1 and sex=='female'"
    ).reset_index()
    df_households_single_woman[column_name] = "Single_wom"

    # get couple without child
    # 2 persons and with couple=True
    df_households_couple = (
        population.query("couple==True")
        .groupby([household_id], as_index=False)
        .agg({"couple": "count", "household_size": "first"})
        .query("couple==2")
    )
    df_households_couple_without_child = df_households_couple.query(
        "couple==2 and household_size==2"
    ).reset_index()
    df_households_couple_without_child[column_name] = "Couple_without_child"

    # get couple with child
    # 3 persons or more, 2 persons in couple, and max age of person non couple < 25
    df_households_child = (
        population.query("age<25 and couple==False")
        .groupby([household_id], as_index=False)
        .agg({person_id: "count"})
        .rename(columns={person_id: "child_count"})
    )
    df_households_couple_with_child = (
        df_households_couple.merge(df_households_child, how="left", on=household_id)
        .query("household_size==child_count+2")
        .reset_index()
    )
    df_households_couple_with_child[column_name] = "Couple_with_child"

    # get single parent
    # 2 persons or more, no one in couple, oldest person = [25-60[, and others age < 25
    df_households_no_couple = population.query("couple==False and household_size>=2").sort_values(
        [household_id, "age"], ascending=[True, False]
    )

    df_households_no_couple_oldest = (
        df_households_no_couple.groupby([household_id], as_index=False)
        .agg({"age": "first", person_id: "first"})
        .rename(columns={"age": "oldest_age", person_id: "oldest_person_id"})
    )

    df_households_no_couple_second_oldest = (
        df_households_no_couple.merge(df_households_no_couple_oldest, how="left")
        .query(f"{person_id}!=oldest_person_id")
        .sort_values([household_id, "age"], ascending=[True, False])
        .groupby([household_id], as_index=False)
        .agg({"age": "first"})
        .rename(columns={"age": "second_oldest_age"})
    )

    df_households_single_parent = df_households_no_couple_oldest.merge(
        df_households_no_couple_second_oldest, how="left", on=household_id
    )
    df_households_single_parent = df_households_single_parent.query(
        "oldest_age>=25 and oldest_age<60 and second_oldest_age<25"
    ).reset_index()
    df_households_single_parent[column_name] = "Single_parent"

    # combine type
    combined = pd.concat(
        [
            df_households_single_man,
            df_households_single_woman,
            df_households_couple_without_child,
            df_households_couple_with_child,
            df_households_single_parent,
        ]
    )[[household_id, column_name]]

    assert combined[household_id].is_unique

    population = population.merge(combined, how="left", on=household_id)
    population[column_name].fillna("complex_hh", inplace=True)

    return population


## functions for reading Filosofi data

filosofi_attributes = [
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


def read_filosofi(filepath: str, year: str, attributes: list, communes=None):
    """
    Fetch income distributions by attribute (age, sex, ...) and commune from Filosofi file

    :param filepath: path to Filosofi excel file (DISP_COM)
    :param year: Filosofi data year
    :param attributes: list of attributes and their modalities, with data required for extraction
    :param communes: optional list of communes to filter

    :return: distributions DataFrame
    """
    # build full list of sheets
    sheet_list = []
    for attribute in attributes:
        sheet_list = sheet_list + [x["sheet"] for x in attribute["modalities"]]

    # read Filosofi excel file
    filosofi_sheets = read_filosofi_excel(filepath, sheet_list)

    # fetch distributions for the given attributes
    distributions = read_filosofi_attributes(filosofi_sheets, year, attributes, communes)

    return distributions


def read_filosofi_excel(filepath: str, sheet_list: list):
    """
    Read list of sheets from Filosofi excel file.

    :param filepath: path to Filosofi excel file (DISP_COM)
    :param sheet_list: list of sheets to be read

    :return: DataFrame indexed by sheet
    """
    excel_df = pd.read_excel(filepath, sheet_name=sheet_list, skiprows=5)
    return excel_df


def read_filosofi_attributes(filosofi_sheets, year, attributes: list, communes=None):
    """
    Read distributions from list of attributes and their modalities in filosofi sheets.

    :param filosofi_sheets: Filosofi excel as DataFrame indexed by sheet
    :param year: Filosofi data year
    :param attributes: list of attributes and their modalities, with data required for extraction
    :param communes: optional list of communes to filter

    :return: distributions DataFrame
    """

    concat_list = []

    # browse modalities for each attribute
    for attribute in attributes:
        for modality in attribute["modalities"]:
            # read distributions from filosofi data
            data = read_distributions_from_filosofi(
                filosofi_sheets,
                year,
                modality["sheet"],
                modality["col_pattern"],
                attribute["name"],
                modality["name"],
                communes,
            )

            concat_list.append(data)

    df = pd.concat(concat_list)

    # Validation
    assert len(attributes) == len(df["attribute"].unique())

    return df[
        [
            "commune_id",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "D7",
            "D8",
            "D9",
            "reference_median",
            "attribute",
            "modality",
        ]
    ]


def read_distributions_from_filosofi(
    filosofi_sheets,
    year: str,
    sheet: str,
    col_pattern: str,
    attribute: str,
    modality: str,
    communes=None,
):
    """


    :param filosofi_sheets: Filosofi excel as DataFrame indexed by sheet
    :param year: Filosofi data year
    :param sheet: sheet name
    :param col_pattern: column pattern
    :param attribute: attribute name
    :param modality: modality name
    :param communes: optional list of communes to filter

    :return: distributions DataFrame
    """

    # Load income distribution
    data = filosofi_sheets[sheet][
        ["CODGEO"]
        + [
            "%sD%d" % (col_pattern, q) + year if q != 5 else col_pattern + "Q2" + year
            for q in range(1, 10)
        ]
    ].copy()
    data.columns = ["commune_id", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
    data.loc[:, "reference_median"] = data["D5"]
    data.loc[:, "modality"] = modality
    data.loc[:, "attribute"] = attribute

    if communes is not None:
        data = data[data["commune_id"].isin(communes)]

    return data
