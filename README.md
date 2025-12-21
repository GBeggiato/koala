# Koala
aka a worse panda(s) ~ Barbascura X !

## Description
minimal, in-memory data processing libraries for csv

provides methods for:
- adding, removing and renaming columns
- filtering
- groupby and aggregate
- save and load CSV files
- left, right, inner join

## Example 

```python

import koala

(
    koala
    .Koala
    .read_csv(some_file)
    .show() # prints to the screen something like:

    # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    #  ----------------------------------------------------------------------- 
    # [5.1, 3.5, 1.4, 0.2, 'setosa']
    # [4.9, 3.0, 1.4, 0.2, 'setosa']
    # [4.7, 3.2, 1.3, 0.2, 'setosa']
    # [4.6, 3.1, 1.5, 0.2, 'setosa']
    # [5.0, 3.6, 1.4, 0.2, 'setosa']
    # [5.4, 3.9, 1.7, 0.4, 'setosa']

    .where(lambda x: x["species"] != "setosa")
    .column_add("NEW_COL", lambda x: x["sepal_width"] * 2)
    .group(
        by="species", 
        aggs=[
            koala.Aggregation(
                name = "MAX_SEP_WIDTH_BY_SPECIES",
                col  = "sepal_width",
                func = koala.AggregationFunc.MAX
            ),
            koala.Aggregation(
                name = "MIN_PET_WIDTH_BY_SPECIES",
                col  = "petal_width",
                func = koala.AggregationFunc.MIN
            ),
        ]
    )
    .column_drop("MAX_SEP_WIDTH_BY_SPECIES")
    .dropna()
    .fillna("a null species", subset=["species"])
    .rename({
        "MIN_PET_WIDTH_BY_SPECIES": "a_better_name"
    })
    .sort(by=["a_better_name"])
    .show()
    .to_csv(out_file)
    .show()
)

```

