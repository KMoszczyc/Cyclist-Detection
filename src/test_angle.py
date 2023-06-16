import math
import numpy as np

def radian_angle_diff(x, y):
    """
    Calculate difference between 2 radian angles in radians
    :param x: angle x in radians
    :param y: angle y in radians
    :return: difference in radians
    """
    return np.abs(min(y - x, y - x + 2 * math.pi, y - x - 2 * math.pi, key=abs))

def vectors_radian_angle_diff(v1, v2):
    """
    Calculate an angle between 2 vectors
    :param v1: Vector1 with a base (0, 0)
    :param v2: Vector2 with a base (0, 0)
    :return: difference in radians
    """

    v_src = (0,0)
    a1 = vector_to_angle(v_src, v1)
    a2 = vector_to_angle(v_src, v2)
    return radian_angle_diff(a1, a2)


def angle_to_vector(angle, length):
    """Calculate a 2D vector from angle and length"""
    x2 = np.cos(angle) * length
    y2 = np.sin(angle) * length
    return int(x2), int(y2)


def vector_to_angle(v1, v2):
    """
    From position vector v1 and dst direction vector v2 calculate the angle of the v2
    :param v1: Vector 1 (x, y) - src position vector
    :param v2: Vector 2 (x, y) - dst vector
    :return: angle of the vector in radians
    """
    x1, y1 = v1
    x2, y2 = v2

    return np.arctan2(y2 - y1, x2 - x1)


def radians_to_degrees(angle):
    """
    Convert radian angle to degrees
    :param angle: angle in radians
    :return:
    """
    return math.degrees(angle)

pi = math.pi

# x = pi*4/5
# y= -pi*4/5

v_src = 0, 0
v1 = -1, 10
v2 = -1, 3

a1 = vector_to_angle(v_src, v1)
a2 = vector_to_angle(v_src, v2)

angle_diff_radian = radian_angle_diff(a1, a2)
angle_diff_degrees = radians_to_degrees(angle_diff_radian)

print('vectors:', v1, v2, 'angles:', a1, a2, 'angle diff:', angle_diff_radian, angle_diff_degrees)