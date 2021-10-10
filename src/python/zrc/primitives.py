import functools
import numpy as np
import plotly.graph_objects as go
from .functools import List
from .geometry import *


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
        return Cartesian3(
            self.x[idx],
            self.y[idx],
            self.z[idx],
        )

    def __repr__(self):
        return f"""{self.__class__.__name__} \
<size: x-{self.x.shape}, y-{self.y.shape}, z-{self.z.shape}>"""

    def __len__(self):
        return len(self.x)

    def __sub__(self, other):
        return self.op_zip(other, np.subtract)

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
        return Cartesian3(*self.core.map(f))

    def flatten(self):
        return self.map(lambda i: np.asarray(i).flatten())

    def reduce(self, f):
        return (
            self
            .flatten()
            .map(lambda i: functools.reduce(f, i))
        )

    def op_zip(self, other, op):
        return self.__class__(
            x=op(self.x, other.x), y=op(self.y, other.y), z=op(self.z, other.z)
        )

    def close_enough_to(self, other):
        result = (
            (self - other)
            .map(lambda i: abs(i) < 0.001)
            .flatten()
            .reduce(np.logical_and)
        )
        return functools.reduce(np.logical_and, [result.x, result.y, result.z])

    def elemnt_wise_map(self, f: list):
        assert len(self) == len(f)
        raise NotImplementedError

    def move_by_mat4(self, mat4):
        return move_cart3_point_by_move_matrix4(self, mat4) 

    def move_by_crystalID(self, crystalID, crystalRC):
        return Cartesian3(
            *move_cart3_point_by_move_matrix4(
                self.flatten(),
                rotation_matrix3_n_move_vector_to_move_matrix4(
                    *get_move_matrix3_n_move_vector_from_df_by_crystalID(crystalID)(
                        crystalRC
                    )
                ),
            )
        )

    def to_numpy(self):
        c3_fd = self.map(lambda i: np.asarray(i).flatten())
        return np.stack([c3_fd.x, c3_fd.y, c3_fd.z], axis=0)

    def to_plotly(self, mode="markers", marker=dict(size=1)):
        c3_fd = self.map(lambda i: np.asarray(i).flatten())
        return go.Scatter3d(
            x=c3_fd.x, y=c3_fd.y, z=c3_fd.z, mode="markers", marker=marker
        )

