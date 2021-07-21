import numpy as np
from .func_combinator
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
    
    Pair(x, Pair(y, Pair(z, None)))
    """
    def __init__(self, core):
        self.core = core

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
        return self.core.car
    
    @property
    def y(self):
        return self.core.cdr.car
    
    @property
    def z(self):
        return self.core.cdr.cdr.car

    def map(self, f):
        raise NotImplementedError

    def hmap(self, f):
        raise NotImplementedError

    def elemnt_wise_map(self, f: list):
        assert len(self) == len(f)
        raise NotImplementedError

    @staticmethod
    def from_xyz(x, y, z):
        assert x.shape == y.shape
        assert y.shape == z.shape
        assert type(x) == type(y)
        assert type(x) == type(z)

        #TODO
        #assert_same_type_n_shape(x, y, z) 

        return Cartesian3(
            Pair(x, Pair(y, Pair(z, None)))
        )
    
    @staticmethod
    def from_one_np_array(arr):
        assert len(arr.shape) == 2
        assert arr.shape[1] == 3

        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]
        return Cartesian3.from_xyz(
            x, y, z
        )

    def to_plotly(self, mode="markers", marker = dict(size = 1)):
        return go.Scatter3d(
            x=self.x.flatten(), 
            y=self.y.flatten(), 
            z=self.z.flatten(), 
            mode="markers", marker=marker
        )
