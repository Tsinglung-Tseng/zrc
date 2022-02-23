import numpy as np
import functools
import plotly.graph_objects as go
from .functools import List


def ypr_to_rotation_matrix(ypr):
    def rotation_matrix_x(angle):
        return np.matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rotation_matrix_y(angle):
        return np.matrix(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rotation_matrix_z(angle):
        return np.matrix(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    return (
        rotation_matrix_x(ypr[0])
        * rotation_matrix_y(ypr[1])
        * rotation_matrix_z(ypr[2])
    )


def ypr_n_move_vector_to_move_matrix4(move_args):
    """
    Usage:

    ypr_n_move_vector_to_move_matrix4([1,2,3,4,5,6])
    """
    yaw, pitch, roll, x, y, z = move_args

    return np.block(
        [
            [ypr_to_rotation_matrix([yaw, pitch, roll]), np.matrix([x, y, z]).T],
            [np.zeros(4)],
        ]
    )


def rotation_matrix3_n_move_vector_to_move_matrix4(rotation_matrix3, move_vector):
    return np.block([[rotation_matrix3, np.matrix(move_vector).T], [np.zeros(4)]])


def move_cart3_point_by_move_matrix4(cart3, move_matrix4):
    return (move_matrix4 * cart3_to_padded_mat(cart3))[:3, :]


def cart3_to_padded_mat(cart3):
    cart3_len = len(cart3)
    cart3_mat = cart3.to_numpy()
    return np.block([[cart3_mat], [np.ones(cart3_len)]])


def get_move_matrix3_n_move_vector_from_df_by_crystalID(crystalID):
    """
    Input crystalRC later because multi systems to handle.

    Usage:
    get_move_matrix3_n_move_vector_from_df_by_crystalID(34)(crystalRC)
    """

    def _get_move_matrix3_n_move_vector_from_df_by_crystalID(crystalID, crystalRC):
        return (
            np.array(crystalRC.iloc[crystalID].R),
            np.array(crystalRC.iloc[crystalID].C),
        )

    return functools.partial(
        _get_move_matrix3_n_move_vector_from_df_by_crystalID, crystalID
    )


class Cartesian3:
    """
    a 3-tuple of zcon.array built with pair:

    list(x, y, z)
    """

    def __init__(self, x, y, z):
        assert x.shape == y.shape
        assert y.shape == z.shape
        assert type(x) == type(y)
        assert type(x) == type(z)

        self.core = List(x, y, z)

    @staticmethod
    def get_from_localPos(fdf):
        return Cartesian3(*fdf[["localPosX", "localPosY", "localPosZ"]].to_numpy().T)

    def __getitem__(self, idx):
        return Cartesian3(self.x[idx], self.y[idx], self.z[idx],)

    def __repr__(self):
        return f"""{self.__class__.__name__} \
<size: x-{self.x.shape}, y-{self.y.shape}, z-{self.z.shape}>"""

    def __len__(self):
        return len(self.x)

    def __add__(self, other):
        return self.op_zip(other, np.add)

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
        apply f along x,y,z.
        """
        return Cartesian3(*self.core.map(f))

    def hmap(self, f, other):
        return Cartesian3(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))

    def flatten(self):
        return self.map(lambda i: np.asarray(i).flatten())

    def reduce(self, f):
        return self.flatten().map(lambda i: functools.reduce(f, i))

    def op_zip(self, other, op):
        return self.__class__(
            x=op(self.x, other.x), y=op(self.y, other.y), z=op(self.z, other.z)
        )

    def concat(self, other):
        return Cartesian3(
            np.concatenate([self.x, other.x]),
            np.concatenate([self.y, other.y]),
            np.concatenate([self.z, other.z]),
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
        raise notimplementederror

    def rotate_ypr(self, rv_ypr):
        x, y, z = [
            np.asarray(i).flatten()
            for i in ypr_to_rotation_matrix(rv_ypr) * self.to_numpy()
        ]
        return Cartesian3(x, y, z)

    def move_by_mat4(self, mat4):
        return move_cart3_point_by_move_matrix4(self, mat4)

    def move_by_crystalid(self, crystalid, crystalrc):
        return cartesian3(
            *move_cart3_point_by_move_matrix4(
                self.flatten(),
                rotation_matrix3_n_move_vector_to_move_matrix4(
                    *get_move_matrix3_n_move_vector_from_df_by_crystalid(crystalid)(
                        crystalrc
                    )
                ),
            )
        )

    def to_numpy(self):
        c3_fd = self.map(lambda i: np.asarray(i).flatten())
        return np.stack([c3_fd.x, c3_fd.y, c3_fd.z], axis=0)

    def to_plotly(self, mode="markers", marker=dict(size=1)):
        c3_fd = self.map(lambda i: np.asarray(i).flatten())
        return go.Scatter3d(x=c3_fd.x, y=c3_fd.y, z=c3_fd.z, mode=mode, marker=marker)


class Segment:
    """
    hits = pd.read_csv('/home/zengqinglong/optical_simu/system_Albira_3_ring_debug/hits_box.csv')
    hits_pos = Cartesian3.from_tuple3s(hits[['posX', 'posY', 'posZ']].to_numpy())
    s_10 = Segment(hits_pos[:10], hits_pos[10:20])
    go.Figure([
        *s_10.to_plotly_line(),
        *s_10.to_plotly_segment(mode='lines+markers', marker=dict(size=3))
    ])
    """

    def __init__(self, fst: Cartesian3, snd: Cartesian3):
        self.fst = fst
        self.snd = snd

    @staticmethod
    def from_listmode(lm):
        return Segment(
            fst=Cartesian3.from_tuple3s(lm[:, :3]),
            snd=Cartesian3.from_tuple3s(lm[:, 3:]),
        )

    def __repr__(self):
        return f"""Pair: <fst: {self.fst}; snd: {self.snd}>"""

    def __getitem__(self, key):
        return self.fmap(lambda i: i[key])

    def fmap(self, func):
        return self.__class__(fst=func(self.fst), snd=func(self.snd))

    def seg_length(self):
        return self.fst.distance_to(self.snd)

    def direct_vector(self):
        return (self.fst - self.snd).fmap(lambda i: i / self.seg_length().numpy())

    def hstack(self):
        return self.fst.concat(self.snd)

    @property
    def middle_point(self):
        return (self.fst + self.snd) / 2

    def distance_to_2p_notation(self, other):
        return np.hstack(
            [self.fst.distance_to(other.fst), self.snd.distance_to(other.snd),]
        )

    def to_listmode(self):
        return np.hstack([self.fst.to_numpy().T, self.snd.to_numpy().T])
        # return np.stack([self.fst.to_matrix(), self.snd.to_matrix()], axis=0)

    def to_sp_line3d(self):
        return FuncArray(
            [
                sp.Line3D(
                    segment.fst.to_sp_point3d()[0], segment.snd.to_sp_point3d()[0]
                )
                for segment in self
            ]
        )

    def to_plotly_line(
        self, line_length=600, mode="lines", marker=dict(size=3), **kvargs
    ):
        lines = Segment(
            self.middle_point + self.direct_vector() * line_length / 2,
            self.middle_point - self.direct_vector() * line_length / 2,
        )
        return lines.to_plotly_segment(mode=mode, marker=marker, **kvargs)

    def to_plotly_segment(self, mode="markers+lines", marker=dict(size=3), **kwargs):
        return [
            p.to_plotly(mode="markers+lines", marker=marker)
            for p in self.fst.hmap(lambda a, b: np.stack([a, b]).T, self.snd)
        ]


class plane:
    def __init__(self, reference_points):
        self.reference_points = reference_points

    def __repr__(self):
        return (
            f"""{self.__class__.__name__} <reference_points: {self.reference_points}>"""
        )

    @property
    def reference_line(self) -> Cartesian3:
        return (self.reference_points[0] - self.reference_points[1]).concat(
            self.reference_points[1] - self.reference_points[2]
        )

    @property
    def norm_vector(self) -> Cartesian3:
        return Cartesian3.from_tuple(
            np.einsum("ijk,j,k->i", LeviCivitaSymbol, *self.reference_line.to_numpy().T)
        )


class Surface:
    def __init__(self, vertices: Cartesian3):
        self.vertices = vertices

    @staticmethod
    def from_xy_size(x_length, y_length):
        vertex_x = np.array(
            [x_length / 2, x_length / 2, -x_length / 2, -x_length / 2], dtype=np.float64
        )
        vertex_y = np.array(
            [y_length / 2, -y_length / 2, y_length / 2, -y_length / 2], dtype=np.float64
        )
        vertex_z = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return Surface(Cartesian3(vertex_x, vertex_y, vertex_z))

    def move(self, *move_vector):
        x, y, z = [np.array([dim]) for dim in move_vector]
        return Surface(self.vertices + Cartesian3(x, y, z))

    def rotate_ypr(self, *rv_ypr):
        return Surface(self.vertices.rotate_ypr(*rv_ypr))

    def to_plotly(self, **marker):
        return go.Mesh3d(
            x=self.vertices.x,
            y=self.vertices.y,
            z=self.vertices.z,
            i=[0, 1],
            j=[1, 2],
            k=[2, 3],
            opacity=0.2,
            # color='rgba(255,0,255, 0.4)',
            color="lightblue",
            **marker,
        )


class Box:
    def __init__(self, surface):
        self.surface = surface

    @staticmethod
    def from_size(x, y, z):
        return Box(
            [
                Surface.from_xy_size(x, y).move(0, 0, z / 2),
                Surface.from_xy_size(x, y).move(0, 0, -z / 2),
                Surface.from_xy_size(x, z)
                .rotate_ypr([np.pi / 2, 0, 0])
                .move(0, y / 2, 0),
                Surface.from_xy_size(x, z)
                .rotate_ypr([np.pi / 2, 0, 0])
                .move(0, -y / 2, 0),
                Surface.from_xy_size(y, z)
                .rotate_ypr([0, 0, np.pi / 2])
                .rotate_ypr([0, np.pi / 2, 0])
                .move(x / 2, 0, 0),
                Surface.from_xy_size(y, z)
                .rotate_ypr([0, 0, np.pi / 2])
                .rotate_ypr([0, np.pi / 2, 0])
                .move(-x / 2, 0, 0),
            ]
        )

    @property
    def vertices(self):
        return Cartesian3.from_matrix(
            tf.concat([s.vertices.to_tensor() for s in self.surface], axis=1)
        )

    def to_plotly(self):
        return [s.to_plotly() for s in self.surface]

    def move(self, *move_vector):
        return Box([s.move(*move_vector) for s in self.surface])

    def rotate_ypr(self, *rv_ypr):
        return Box([s.rotate_ypr(*rv_ypr) for s in self.surface])
