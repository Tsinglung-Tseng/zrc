import numpy as np
import functools


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

    return np.block([
        [ypr_to_rotation_matrix([yaw, pitch, roll]), np.matrix([x,y,z]).T],
        [np.zeros(4)]
    ])

def rotation_matrix3_n_move_vector_to_move_matrix4(rotation_matrix3, move_vector):
    return np.block([
        [rotation_matrix3, np.matrix(move_vector).T],
        [np.zeros(4)]
    ])

def move_cart3_point_by_move_matrix4(cart3, move_matrix4):    
    return (
        move_matrix4
        * 
        cart3_to_padded_mat(cart3)
    )[:3,:]

def cart3_to_padded_mat(cart3):
    cart3_len = len(cart3)
    cart3_mat = cart3.to_numpy()
    return np.block([
        [cart3_mat],
        [np.ones(cart3_len)]
    ])

def get_move_matrix3_n_move_vector_from_df_by_crystalID(crystalID):
    """
    Input crystalRC later because multi systems to handle.

    Usage:
    get_move_matrix3_n_move_vector_from_df_by_crystalID(34)(crystalRC)
    """
    def _get_move_matrix3_n_move_vector_from_df_by_crystalID(crystalID, crystalRC):
        return np.array(crystalRC.iloc[crystalID].R), np.array(crystalRC.iloc[crystalID].C)
    return functools.partial(_get_move_matrix3_n_move_vector_from_df_by_crystalID, crystalID)

