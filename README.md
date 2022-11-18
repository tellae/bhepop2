# synthetic-pop-uge-tellae

```mermaid
flowchart TD
    fil["raw Filosofi\n(INSEE xlsx)"] --> rf[/"read_filosofi()"/]
    rf --> income["Raw income\nDataFrame"]
    pop["Population EQASIM\nDataFrame"] --> ah[/"add_attributes_households()"/]
    hsd["Households EQASIM\nDataFrame"] --> ah
    ah --> hsdnew["Households DataFrame\n with new attributes"]
    incomeimputed["Imputed income EQASIM"] --> opt
    income --> opt
    hsdnew --> opt[/"optimise()"/]

```

- `read_filosofi()` : TODO what is does, describe input data, describe output data
- `add_attributes_households()` : TODO what it does, describe input data, describe output data
- `optimise()`
