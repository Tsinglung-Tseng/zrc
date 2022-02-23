import functools
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
