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

def calculate_absolute_diff_for_angles(self, angle1, angle2):
    """
    Calculate Mean Absolute Error between 2 angles
    :param x: angle x in radians
    :param y: angle y in radians
    :return: difference in radians
    """
    return min([angle2 - angle1, angle2 - angle1 + 2 * math.pi, angle2 - angle1 - 2 * math.pi], key=abs)
def calculate_squared_error_for_angles(angle1, angle2):
    """
    Calculate Root Mean Squared Error between 2 angles
    :param x: angle x in radians
    :param y: angle y in radians
    :return: difference in radians
    """
    return min((angle2 - angle1) ** 2, (angle2 - angle1 + 2 * math.pi) ** 2, (angle2 - angle1 - 2 * math.pi) ** 2, key=abs)

def squared_error_deg(angle1, angle2):
    angle1_deg = math.degrees(angle1)
    angle2_deg = math.degrees(angle2)
    return min((angle2_deg - angle1_deg) ** 2, (angle2_deg - angle1_deg + 360) ** 2, (angle2_deg - angle1_deg - 360) ** 2, key=abs)

l1 = [(10, 10), (-3, 5), (-1, 3)]
l2 = [(-3, 5), (-2, 5), (1, 2)]

pi = math.pi

# x = pi*4/5
# y= -pi*4/5

v_src = 0, 0
v1 = -1, 10
v2 = -1, 3

mae_list = []
rmse_list = []
rmse_degrees_list = []

for v1, v2 in zip(l1, l2):
    a1 = vector_to_angle(v_src, v1)
    a2 = vector_to_angle(v_src, v2)

    mae_radian = radian_angle_diff(a1, a2)
    rmse_radian = calculate_squared_error_for_angles(a1, a2)
    rmse_degrees = squared_error_deg(a1, a2)

    mae_list.append(mae_radian)
    rmse_list.append(rmse_radian)
    rmse_degrees_list.append(rmse_degrees)

mae_radian_avg = sum(mae_list)/len(mae_list)
rmse_radian_avg = np.sqrt(sum(rmse_list)/len(rmse_list))
rmse_degrees_avg_v2 = np.sqrt(sum(rmse_degrees_list)/len(rmse_list))


mae_degrees_avg = radians_to_degrees(mae_radian_avg)
rmse_degrees_avg = radians_to_degrees(rmse_radian_avg)

print('mae:', mae_radian_avg, mae_degrees_avg, 'rmse:', rmse_radian_avg, rmse_degrees_avg, 'rmse_degrees_avg_v2', rmse_degrees_avg_v2)

xs = [10, 1, 3, 10, 4, 6, 12, 1]
print('std', np.std(xs))

l = [1, 2, 3, 4, 5, 6, 7]
print(l[-3:])