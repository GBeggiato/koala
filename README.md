
# Koala

aka a worse panda(s)

## Description

minimal, in-memory data processing libraries for csv

provides methods for adding and removing columns, filtering, groupby and join, 
save and load to CSV.

## Example 

```python

from koala import Koala, AggregationFunc

(
    Koala
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
    .group(
        by="species", 
        aggs=[
            ("MAX_SEP_WIDTH_BY_SPECIES", "sepal_width", AggregationFunc.MAX),
            ("MIN_PET_WIDTH_BY_SPECIES", "petal_width", AggregationFunc.MIN),
        ]
    )
    .column_drop("MAX_SEP_WIDTH_BY_SPECIES")
    .dropna()
    .fillna(0, subset=["species"])
    .join_left(Koala.read_csv(f), on=["species"])
    .join_inner(Koala.read_csv(f), on=["species"])
    .rename({
        "MIN_PET_WIDTH_BY_SPECIES": "a_better_name"
    })
    .sort(by=["a_better_name"])
    .show()
    .to_csv(out_file)
)

```
