from functools import reduce
import operator as op
import numpy as np
import plotly.graph_objects as go
from collections.abc import Iterable


def identical(fst, snd):
    """
    Check if all items in list identical.

    reduce(identical, map_list(type, [1,2,3]))
    >>> (True, int)
    """
    if not isinstance(fst, tuple):
        return (fst==snd, snd)
    else:
        last_result, last_item = fst
        this_result = op.and_(last_result, last_item==snd)
        return (this_result, snd)


map_list = lambda f, l: list(map(f, l))


unified_items = lambda l: reduce(identical, map_list(type, l))[0]


class Pair:
    """
    Any sequential pairs that can be processed in parallel.

    Attributes
    ----------
    car : <T>
        first element of pair
    cdr : <T>
        second element of pair

    Methods
    -------
    map(f=procedure):
        Do the same thing to car/cdr by mapping a procedure to car and cdr.
    """
    def __init__(self, car, cdr):
        self.car=car
        self.cdr=cdr

    def map(self, f):
        """
        Pair(1,2).map(double) => Pair(2,4)

        Parameters
        ----------
            f : procedure
        """
        if type(self.cdr)==Pair:
            return Pair(f(self.car), self.cdr.map(f))
        elif self.cdr is not None:
            return Pair(f(self.car), f(self.cdr))
        else:
            return Pair(f(self.car), None)


class List:
    """
    A list constructed with Pair List(1,2,3,4)==Pair(1, Pair(2, Pair(3, Pair(4, None)))).

    List(1,2,3,4).core.map(lambda x:x*2).cdr.cdr.cdr.car
    >>> 8
    """
    def __init__(self, *args):
        def make_list(l):
            assert unified_items(l)
            first, *rest = l

            if len(rest)==1:
                return Pair(first, Pair(rest[0], None))
            else:
                return Pair(first, make_list(rest))
        
        if isinstance(args[0], Pair):
            self.core = args[0]
        elif isinstance(args, Iterable):
            self.core = make_list(args)
    
    def __getitem__(self, idx):
        if idx==0:
            return self.core.car
        else:
            try:
                return List(self.core.cdr)[idx-1]
            except TypeError:
                raise IndexError("Index out of bound.")

    def map(self, f):
        return List(self.core.map(f))


class FuncDataFrame:
    """
    A functional pandas dataframe wrapper.
    
    hits = FuncDataFrame(pd.read_csv("route/to/hits.csv"))
    hits.where(processName="Compton").count()
    >>> 123
    """
    def __init__(self, df):
        self.df = df
    
    def __getattr__(self, attr):
        """
        Get inner pandas dataframe directly, when attr is not seen in wrapper function,
        designed only for fast access pandas dataframe columns.
        """
        return getattr(self.df, attr)

    def _pass_func_to_inner_df(self, f):
        """Pass procedures to inner pandas dataframe of FuncDataFrame."""
        return f(self.df)
    
    def count(self):
        return self.shape[0]
        
    def select_where(self, **kwargs):
        """
        hits.select_where(processName="Compton")
        """
        if len(kwargs) != 1:
            raise ValueError("where clause support one condition at once!")
        for key, value in kwargs.items():
            return FuncDataFrame(self.df[self.df[key] == value])

    def select(self, labels):
        return FuncDataFrame(self.df[labels])

    def to_numpy(self):
        return self.df.to_numpy()

