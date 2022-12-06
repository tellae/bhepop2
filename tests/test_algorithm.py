from src import functions
from src.tools import *
from src.nantes_application import PATH_INPUTS, SYNTHETIC_POP, CODE_INSEE, MODALITIES
import pandas as pd
import numpy as np


def test_algorithm():

    synth_pop = pd.read_csv(PATH_INPUTS + SYNTHETIC_POP, sep=";")
    crossed_probabilities = functions.compute_crossed_probabilities(synth_pop, MODALITIES)

    # %%
    # script 2 : read data incomes distribution from xls INSEE raw data
    df_income_attributes = read_filosofi(
        "../data/raw/indic-struct-distrib-revenu-2015-COMMUNES/FILO_DISP_COM.xls"
    ).query(f"commune_id == '{CODE_INSEE}'")

    filosofi = df_income_attributes.copy()
    filosofi = filosofi[filosofi["attribute"].isin(functions.get_attributes(MODALITIES))]
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

    vec_all_incomes = functions.compute_vec_all(filosofi)
    p_R = functions.compute_p_r(vec_all_incomes, df_income_attributes, CODE_INSEE)

    assert list(p_R.columns) == ["income", "proba1"]
    assert len(p_R) == 190
    assert p_R.iloc[-1, 1] == 1.0

    res = functions.run_assignment(filosofi, vec_all_incomes, crossed_probabilities, MODALITIES)

    expected = pd.read_csv("../tests/nantes_result.csv")
    expected.columns = [int(x) for x in expected.columns]

    assert np.all(np.isclose(expected.to_numpy(), res.to_numpy()))

    for i in range(7):
        res[i] = res[i] * p_R["proba1"][i]
    res["sum"] = res.sum(axis=1)
    for i in range(7):
        res[i] = res[i] / res["sum"]
    res["sum"] = res.sum(axis=1)

    # print(res[round(res["sum"], 1) != 1.00])
    assert (round(res["sum"]) == 1.00).all()
