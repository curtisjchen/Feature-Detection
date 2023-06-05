import math
import numpy as np


def get_rot_mx(angle_z):
    '''
    Input:
        angle_x -- Rotation around the x axis in radians
        angle_y -- Rotation around the y axis in radians
        angle_z -- Rotation around the z axis in radians
    Output:
        A 3x3 numpy array representing 2D rotations. 
    '''
    # Note: For MOPS, you need to use angle_z only, since we are in 2D

    #rot_x_mx = np.array([[1, 0, 0, 0],
    #                     [0, math.cos(angle_x), -math.sin(angle_x), 0],
    #                     [0, math.sin(angle_x), math.cos(angle_x), 0],
    #                     [0, 0, 0, 1]])

    #rot_y_mx = np.array([[math.cos(angle_y), 0, math.sin(angle_y), 0],
    #                     [0, 1, 0, 0],
    #                    [-math.sin(angle_y), 0, math.cos(angle_y), 0],
    #                     [0, 0, 0, 1]])

    rot_z_mx = np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z), math.cos(angle_z), 0],
                         [0, 0, 1]])

    return rot_z_mx


def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 2
        elements
    Output:
        A 3x3 numpy array representing 2D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    trans_mx = np.eye(3)
    trans_mx[0:2, 2] = trans_vec

    return trans_mx


def get_scale_mx(s_x, s_y):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
    Output:
        A 3x3 numpy array representing 2D scaling.
    '''
    scale_mx = np.eye(3)

    for i, s in enumerate([s_x, s_y]):
        scale_mx[i, i] = s

    return scale_mx

