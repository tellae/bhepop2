from bhepop2.tools import add_size_attribute


def test_add_size_attribute(eqasim_population, eqasim_households):
    pop = add_size_attribute(eqasim_population)
    assert set(pop["size"].unique()) == {"1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"}

