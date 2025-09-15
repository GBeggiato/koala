from __future__ import annotations

import collections
import copy
import csv
import dataclasses
import enum
import itertools
from pathlib import Path
import statistics
from sys import implementation
import typing as ty


type Row = dict[str, ty.Any]
type Func[T] = ty.Callable[[Row], T]
type StrS = str | list[str]


class _JoinKind(enum.Enum):
    INNER = enum.auto()
    LEFT = enum.auto()


def _listify(s: StrS) -> list[str]:
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


# TODO: map more functions + add custom
class AggregationFunc(enum.Enum):
    SUM = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()
    MEAN = enum.auto()
    STD = enum.auto()
    MEDIAN = enum.auto()
    COUNT = enum.auto()
    FIRST = enum.auto()

    def _callable(self) -> ty.Callable:
        msg = "bad aggregation function provided"
        if self == AggregationFunc.SUM    : return sum
        if self == AggregationFunc.MIN    : return min
        if self == AggregationFunc.MAX    : return max
        if self == AggregationFunc.MEAN   : return statistics.mean
        if self == AggregationFunc.STD    : return statistics.stdev
        if self == AggregationFunc.MEDIAN : return statistics.median
        if self == AggregationFunc.COUNT  : return len
        if self == AggregationFunc.FIRST  : return lambda x: x[0]
        raise ValueError(msg)


@dataclasses.dataclass(slots=True)
class Koala:
    _cols: list[str]
    _rows: list[list]

    @property
    def columns(self) -> list[str]:
        return self._cols

    @property
    def shape(self) -> tuple[int, int]:
        """(n cols, n rows)"""
        return len(self._cols), len(self._rows)

    def _col_index(self, c: str) -> int:
        try:
           return self._cols.index(c)
        except ValueError:
            raise ValueError(f"column {c} not found")

    def clone(self) -> Koala:
        """deepcopy"""
        return copy.deepcopy(self)

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
    def read_csv(cls, f: Path) -> Koala:
        with f.open() as fp:
            lines = iter(csv.reader(fp))
            return cls(
                list(next(lines)), 
                list(list(map(_parse, row)) for row in lines)
            )

    def get(self, col: str) -> list:
        i = self._col_index(col)
        return [r[i] for r in self._rows]

    def _row_as_dict(self, row: list) -> Row:
        return dict(zip(self._cols, row))

    def as_dicts(self) -> ty.Generator[Row, ty.Any, None]:
        yield from map(self._row_as_dict, self._rows)

    def where(self, f: Func[bool]) -> Koala:
        """filter rows according to a provided predicate"""
        self._rows = [r for r in self._rows if f(self._row_as_dict(r))]
        return self

    def column_drop(self, col: str) -> Koala:
        """drop a provided column, fails if col is not there"""
        position = self._col_index(col)
        del self._cols[position]
        for row in self._rows:
            del row[position]
        return self

    def column_add(self, name: str, f: Func[ty.Any]) -> Koala:
        """
        add a column based on a predicate.
        if an existing name is provided, the
        old column is dropped
        """
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
                if groups.get(result_name) is None:
                    groups[result_name] = collections.defaultdict(list)
                groups[result_name][key].append(row[aggregated_col])
        return groups

    @staticmethod
    def _agg(by: StrS, aggs: list[tuple[str, str, AggregationFunc]], groups: dict) -> tuple[list, list]:
        cols = []
        rows = collections.defaultdict(list)
        for i, agg in enumerate(aggs):
            result_name, _, agg_func = agg
            f = agg_func._callable()
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

    def group(self, by: StrS, aggs: list[tuple[str, str, AggregationFunc]]) -> Koala:
        """
        group by + aggregate

        ## Example
        ```python3
        .group(
            by=["key"], # these go in a list
            aggs=[( # and you need a list of these
                    "tot_value_by_aggr", # the name of the new column
                    "column",            # the column we are aggregating
                    AggregationFunc.SUM  # the function we are using to aggregate
            )]
        )
        ```
        """
        groups = self._groupby(by, aggs)
        cols, rows = self._agg(by, aggs, groups)
        self._cols = _flatten(cols)
        self._rows = list(map(_flatten, rows))
        return self

    def sort(self, by: StrS, reverse: bool = False) -> Koala:
        """sort according to column value"""
        ii = [self._col_index(c) for c in by]
        def _sort_fn(x) -> list:
            return [x[i] for i in ii]
        self._rows.sort(key=_sort_fn, reverse=reverse)
        return self

    def rename(self, renamer: dict[str, str]):
        """rename columns"""
        self._cols = [renamer.get(k, k) for k in self._cols]
        return self

    def _has_col(self, col: str) -> bool:
        return col in self._cols

    def dropna(self, subset: ty.Optional[list[str]] = None) -> Koala:
        """drop None values on an optional subset of cols (or all of them)"""
        if subset is None:
            def keep(r: list) -> bool:
                return not (None in self._row_as_dict(r).values())
        else:
            idxs = [self._col_index(c) for c in subset]
            def keep(r: list) -> bool:
                return all(r[i] is not None for i in idxs)
        self._rows = list(filter(keep, self._rows))
        return self

    def fillna(self, value: ty.Any, subset: ty.Optional[list[str]]=None):
        """fill None values on an optional subset of cols (or all of them)"""
        idxs = range(len(self._cols)) if subset is None else [
            self._col_index(c) for c in subset
        ]
        indices = itertools.product(range(len(self._rows)), idxs)
        for r, i in indices:
            if self._rows[r][i] is None:
                self._rows[r][i] = value
        return self

    def to_csv(self, f: Path) -> Koala:
        with f.open("w") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._cols)
            writer.writeheader()
            writer.writerows(self.as_dicts())
        return self

    @classmethod
    def from_dict_list(cls, ds: list[Row]) -> Koala:
        _cols = list(next(iter(ds)).keys())
        _rows = [[r[k] for k in _cols] for r in ds]
        return cls(_cols, _rows)

    def join_left(self, right: Koala, join_key: StrS):
        return self._join(self, right, join_key, _JoinKind.LEFT, dict())

    def join_right(self, left: Koala, join_key: StrS):
        return self._join(left, self, join_key, _JoinKind.LEFT, dict())

    def join_inner(self, right: Koala, join_key: StrS):
        return self._join(self, right, join_key, _JoinKind.INNER, None)

    @staticmethod
    def _join(
        left     : Koala,
        right    : Koala,
        join_key : StrS,
        kind     : _JoinKind,
        default  : ty.Any
    ) -> Koala:
        join_key = _listify(join_key)

        def _join_key(d: dict) -> int:
            return hash(tuple(d[k] for k in join_key))

        left = left.clone().rename({c: f"{c}_left" for c in left._cols if c not in join_key})
        right = right.clone().rename({c: f"{c}_right" for c in right._cols if c not in join_key})
        cols = set(left._cols).union(right._cols)
        right_dict = collections.defaultdict(list)
        for r in right.as_dicts():
            right_dict[_join_key(r)].append(r)
        result = []
        for left_row in left.as_dicts():
            right_row = right_dict.get(_join_key(left_row), default)
            if kind == _JoinKind.INNER and right_row is None:
                continue
            assert right_row is not None
            for rr in right_row:
                row = copy.deepcopy(left_row)
                row.update(rr)
                result.append({c: row.get(c) for c in cols})
        return Koala.from_dict_list(result)

