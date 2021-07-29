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

