import numpy as np
import plotly.graph_objects as go


class Array:
    def __init__(self, core):
        assert type(core) == np.ndarray
        self.core = core

    def map(self, f):
        return Array(np.vectorize(f)(self.core))

    def to_numpy(self):
        return self.core


class Pair:
    def __init__(self, car, cdr):
        self.car=car
        self.cdr=cdr

    def map(self, f):
        if type(self.cdr)==Pair:
            return Pair(f(self.car), self.cdr.map(f))
        else:
            return Pair(f(self.car), None)
        
class Cart3:
    """
    A 3-tuple of zcon.array built with Pair:
    
    Pair(x, Pair(y, Pair(z, None)))
    """
    def __init__(self, core):
        self.core = core

    def __getitem__(self, idx):
        return Cart3.from_xyz(
            self.x[idx],
            self.y[idx],
            self.z[idx],
        )
        
    def __repr__(self):
        return f"""{self.__class__.__name__} \
<size: x-{self.x.shape}, y-{self.y.shape}, z-{self.z.shape}>"""

    @property
    def x(self):
        return self.core.car
    
    @property
    def y(self):
        return self.core.cdr.car
    
    @property
    def z(self):
        return self.core.cdr.cdr.car

    @staticmethod
    def from_xyz(x, y, z):
        assert x.shape == y.shape
        assert y.shape == z.shape
        assert type(x) == type(y)
        assert type(x) == type(z)
        return Cart3(
            Pair(x, Pair(y, Pair(z, None)))
        )
    
    @staticmethod
    def from_one_np_array(arr):
        assert len(arr.shape) == 2
        assert arr.shape[1] == 3
        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]
        return Cart3.from_xyz(
            x, y, z
        )

    def to_plotly(self, mode="markers", marker = dict(size = 1)):
        return go.Scatter3d(
            x=self.x.flatten(), 
            y=self.y.flatten(), 
            z=self.z.flatten(), 
            mode="markers", marker=marker
        )
