import numpy as np
import plotly.graph_objects as go
from .functools import List


class Array:
    def __init__(self, core):
        assert type(core) == np.ndarray
        self.core = core

    def __repr__(self):
        return f"""{self.__class__.__name__} \
<{self.core}>"""

    def map(self, f):
        return Array(np.vectorize(f)(self.core))

    def to_numpy(self):
        return self.core

    def to_tf_tensor(self):
        raise NotImplementedError

    def to_torch_tensor(self):
        raise NotImplementedError


class Cartesian3:
    """
    A 3-tuple of zcon.array built with Pair:
   
    List(x, y, z)
    """
    def __init__(self, x, y, z):
        assert x.shape == y.shape
        assert y.shape == z.shape
        assert type(x) == type(y)
        assert type(x) == type(z)

        self.core = List(x, y, z)

    def __getitem__(self, idx):
        return Cartesian3.from_xyz(
            self.x[idx],
            self.y[idx],
            self.z[idx],
        )
        
    def __repr__(self):
        return f"""{self.__class__.__name__} \
<size: x-{self.x.shape}, y-{self.y.shape}, z-{self.z.shape}>"""

    def __len__(self):
        return len(self.x)
    
    @property
    def x(self):
        return self.core[0]
    
    @property
    def y(self):
        return self.core[1]
    
    @property
    def z(self):
        return self.core[2]

    def map(self, f):
        """
        Apply f along x,y,z.
        """
        result = self.core.map(f)
        return Cartesian3(
            result[0],
            result[1],
            result[2],
        )

    # def hmap(self, f):
        # """
        # Apply f to every Cartesian3.
        # """
        # result = np.apply_along_axis(f, 0, self.to_numpy())
        # return Cartesian3(
            # result[0,:],
            # result[1,:],
            # result[2,:],
        # ) 

    def elemnt_wise_map(self, f: list):
        assert len(self) == len(f)
        raise NotImplementedError

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)
    
    def to_plotly(self, mode="markers", marker = dict(size = 1)):
        c3_fd = self.map(lambda coord: coord.flatten())
        return go.Scatter3d(
            c3_fd.x, 
            c3_fd.y, 
            c3_fd.z, 
            mode="markers", marker=marker
        )
