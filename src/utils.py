"""
Utility functions
"""

import pandas as pd


def interpolate_income(income: float, distribution: list):
    """
    Linear interpolation of incomes

    :param income: value of income to interpolate
    :param distribution: list of incomes for each decile from 1 to 10 (without value for 0)
    :return: probability of being lower than income value
    """
    distribution = [0] + distribution
    if income > distribution[10]:
        return 1
    if income < distribution[0]:
        return 0
    decile_top = 0
    while income > distribution[decile_top]:
        decile_top += 1

    interpolation = (income - distribution[decile_top - 1]) * (
        decile_top * 0.1 - (decile_top - 1) * 0.1
    ) / (distribution[decile_top] - distribution[decile_top - 1]) + (decile_top - 1) * 0.1

    return interpolation


def read_filosofi(
    path: str, sheet: str, code_insee: str, xls_file="FILO_DISP_COM.xls", skip_rows=5
):
    """
    Read Filosofi data from raw xls file

    :param path: full path to directory containing xls file
    :param sheet: sheet name where to get data
    :param code_insee: insee code of selected municipality
    :param xls_file: name of xls file
    :param skip_rows: number of rows to skip in sheet
    :return: data frame of data
    """
    data_frame = pd.read_excel(
        path + xls_file,
        sheet_name=sheet,
        skiprows=skip_rows,
    ).query("CODGEO=='" + code_insee + "'")

    return data_frame
