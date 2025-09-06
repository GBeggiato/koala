from __future__ import annotations

import collections
import csv
import dataclasses
import enum
import itertools
from pathlib import Path
import statistics
import typing as ty


type Row = dict[str, ty.Any]
type Func[T] = ty.Callable[[Row], T]
type StrS = str | list[str]


def _listify(s: str | list[str]) -> list[str]:
    if isinstance(s, str):
        return [s]
    return s


def _flatten(xs: list | tuple):  
    res = []  
    for x in xs:  
        if isinstance(x, (list, tuple)):  
            res.extend(_flatten(x))
        else:  
            res.append(x)
    return res  


def _parse(x: str) -> float | str:
    if x.replace(".", "").isdigit():
        return float(x)
    return x


def _has_no_null_values(x: dict) -> bool:
    return not (None in x.values())


# TODO: map more functions + add custom
class AggregationFunc(enum.Enum):
    SUM = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()
    MEAN = enum.auto()
    STD = enum.auto()
    MEDIAN = enum.auto()
    COUNT = enum.auto()

    @staticmethod
    def _get_agg_func(s) -> ty.Callable:
        msg = "bad aggregation function provided"
        if not isinstance(s, AggregationFunc):
            raise ValueError(msg)
        if s == AggregationFunc.SUM: return sum
        if s == AggregationFunc.MIN: return min
        if s == AggregationFunc.MAX: return max
        if s == AggregationFunc.MEAN: return statistics.mean
        if s == AggregationFunc.STD: return statistics.stdev
        if s == AggregationFunc.MEDIAN: return statistics.median
        if s == AggregationFunc.COUNT: return len
        raise ValueError(msg)


@dataclasses.dataclass(slots=True)
class Koala:
    _cols: list[str]
    _rows: list[list]

    def show(self, n:int=6):
        print(self._cols)
        c = len(str(self._cols)) - 2
        print(" "+("-"*c)+" ")
        for i, e in enumerate(self._rows):
            if i < n:
                print(e)
        print()
        return self

    @classmethod
    def read_csv(cls, f: Path) -> ty.Self:
        with f.open() as fp:
            lines = iter(csv.reader(fp))
            return cls(
                list(next(lines)), 
                list(list(map(_parse, row)) for row in lines)
            )

    def _row_as_dict(self, row: list) -> Row:
        return dict(zip(self._cols, row))

    def _rows_as_dicts(self):
        yield from map(self._row_as_dict, self._rows)

    def where(self, f: Func[bool]) -> ty.Self:
        self._rows = [r for r in self._rows if f(self._row_as_dict(r))]
        return self

    def column_drop(self, col: str) -> ty.Self:
        position = self._cols.index(col)
        del self._cols[position]
        for row in self._rows:
            del row[position]
        return self

    def column_add(self, name: str, f: Func[ty.Any]) -> ty.Self:
        drop_old = name in self._cols
        self._cols.append(name)
        for row in self._rows:
            row.append(f(self._row_as_dict(row)))
        if drop_old:
            self.column_drop(name)
        return self

    @staticmethod
    def _get_group_key(by: StrS, row: Row) -> str | tuple:
        if isinstance(by, str):
            return row[by] 
        return tuple(row[b] for b in by)

    def _groupby(self, by: StrS, aggs: list[tuple[str, str, AggregationFunc]]) -> dict:
        groups = dict()
        rows = map(self._row_as_dict, self._rows)
        for row in rows:
            key = self._get_group_key(by, row)
            for agg in aggs:
                result_name, aggregated_col, _ = agg
                if result_name not in groups:
                    groups[result_name] = collections.defaultdict(list)
                groups[result_name][key].append(row[aggregated_col])
        return groups

    def _agg(self, by: StrS, aggs: list[tuple[str, str, AggregationFunc]], groups: dict) -> tuple[list, list]:
        cols = []
        rows = collections.defaultdict(list)
        for i, agg in enumerate(aggs):
            result_name, _, agg_func = agg
            f = AggregationFunc._get_agg_func(agg_func)
            if i == 0:
                cols.append(by)
            cols.append(result_name)
            group = groups[result_name]
            for k, v in group.items():
                rows[k].append(f(v))
        new_rows = []
        for k, v in rows.items():
            row = [k]
            row.extend(v)
            new_rows.append(row)
        return cols, new_rows

    def group(self, by: StrS, aggs: list[tuple[str, str, AggregationFunc]]) -> ty.Self:
        groups = self._groupby(by, aggs)
        cols, rows = self._agg(by, aggs, groups)
        self._cols = _flatten(cols)
        self._rows = list(map(_flatten, rows))
        return self

    def sort(self, by: StrS, reverse: bool = False) -> ty.Self:
        
        def _sort_fn(x) -> list:
            return [x[self._cols.index(c)] for c in by]

        self._rows.sort(key=_sort_fn, reverse=reverse)
        return self

    def rename(self, renamer: dict[str, str]):
        self._cols = [renamer.get(k, k) for k in self._cols]
        return self

    def _has_col(self, col: str) -> bool:
        return col in self._cols

    # def _join(self, right: ty.Self, on: StrS):
    #     """
    #     it's the user's responsibility to have the join
    #     columns be the same on both tables
    #     """
    #     on = _listify(on)
    #     left_has_keys = all(self._has_col(c) for c in on)
    #     right_has_keys = all(right._has_col(c) for c in on)
    #     if not (left_has_keys and right_has_keys):
    #         raise KeyError(f"not a valid join key: {on}")
    #
    #     left_only = set(c for c in self._cols if c not in on)
    #     right_only = set(c for c in right._cols if c not in on)
    #     common = left_only.intersection(right_only)
    #     if common:
    #         self.rename({c: f"{c}_left" for c in common})
    #         right.rename({c: f"{c}_right" for c in common})
    #
    #     lefts = list(map(self._row_as_dict, self._rows))
    #     rights = list(map(right._row_as_dict, right._rows))
    #     for l in lefts:
    #         for r in rights:
    #             if all(r[k] == l[k] for k in on):
    #                 l.update(r)
    #     return lefts
    #
    # def join_left(self, right: ty.Self, on: StrS) -> Koala:
    #     cols = self._cols
    #     cols.extend((c for c in right._cols if c not in cols))
    #     rows = [[r.get(k) for k in cols] for r in self._join(right, on)]
    #     return Koala(cols, rows)
    #
    # def join_inner(self, right: ty.Self, on: StrS) -> Koala:
    #     return self.join_left(right, on).where(_has_no_null_values)
    #
    # def join_outer(self):
    #     pass
    #
    # def join_cross(self):
    #     pass

    def dropna(self, subset: ty.Optional[list[str]] = None) -> ty.Self:

        if subset is None:
            def keep(r: list) -> bool:
                return _has_no_null_values(self._row_as_dict(r))
        else:
            idxs = [self._cols.index(c) for c in subset]
            def keep(r: list) -> bool:
                return all(r[i] is not None for i in idxs)

        self._rows = list(filter(keep, self._rows))
        return self

    def fillna(self, value: ty.Any, subset: ty.Optional[list[str]]=None):
        idxs = range(len(self._cols)) if subset is None else [
            self._cols.index(c) for c in subset
        ]
        indices = itertools.product(range(len(self._rows)), idxs)
        for r, i in indices:
            if self._rows[r][i] is None:
                self._rows[r][i] = value
        return self

    def to_csv(self, f: Path) -> ty.Self:
        with f.open("w") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._cols)
            writer.writeheader()
            writer.writerows(self._rows_as_dicts())
        return self















#
# def check():
#     import pandas as pd
#
#     df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
#     df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
#
#     print(df1)
#     print(df2)
#     print(df1.merge(df2, left_on='lkey', right_on='rkey'))
#
#     cols = ["key", "value"]
#     rows = [["foo", 1],["bar", 2],["baz", 3],["foo", 5]]
#     k1 = Koala(cols, rows)
#
#     cols = ["key", "value"]
#     rows = [["foo", 5],["bar", 6],["baz", 7],["foo", 8]]
#     k2 = Koala(cols, rows)
#
#     k1.show()
#     k2.show()
#     k1.join_left(k2, on=["key"]).show()
#
# if __name__ == "__main__":
#     check()
#
#
#
