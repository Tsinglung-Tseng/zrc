import numpy as np
import plotly.graph_objects as go


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
        elif self.cdr is None:
            return Pair(f(self.car), None)
        else:
            return Pair(f(self.car), f(self.cdr))
        

class List:
    def __init__(self, *args):
        raise NotImplementedError
