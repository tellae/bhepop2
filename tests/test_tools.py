from bhepop2.tools import (
    add_household_size_attribute,
    add_household_type_attribute,
)


def test_add_household_size_attribute(eqasim_population):
    pop = add_household_size_attribute(eqasim_population)
    assert set(pop["size"].unique()) == {"1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"}


def test_add_household_type_attribute(eqasim_population):
    pop = add_household_type_attribute(eqasim_population)
    assert set(pop["family_comp"].unique()) == {
        "Single_man",
        "Single_wom",
        "Couple_without_child",
        "Couple_with_child",
        "Single_parent",
        "complex_hh",
    }
