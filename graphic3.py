import numpy as np
import quaternion
import glfw
from OpenGL.GL import *
import sys

rng = np.random.default_rng()

def rotate_with_planeP(P, v, theta):
    """
    :param P: (vector, vector)
    :param v: vector
    :param theta: angle
    :return: vector
    """
    p1 = P[0]
    p2 = P[1]
    q_v = quaternion.quaternion(v[0], v[1], v[2], v[3])
    v_parallel = np.dot(v, p1) * p1 + np.dot(v, p2) * p2 #parallel:平行
    if np.all(v_parallel == 0):
        v_parallel = (0, 0, 0, 0)
    q_v_parallel = quaternion.quaternion(v_parallel[0], v_parallel[1], v_parallel[2], v_parallel[3])
    q_p1 = quaternion.quaternion(p1[0], p1[1], p1[2], p1[3])
    q_p2 = quaternion.quaternion(p2[0], p2[1], p2[2], p2[3])
    q_v_vertical = q_v - q_v_parallel
    ans_q_v = q_v_parallel + (q_p1 * np.cos(theta) + q_p2 * np.sin(theta)) * q_v_vertical
    ans_v = np.array([ans_q_v.w, ans_q_v.x, ans_q_v.y, ans_q_v.z], dtype=np.float64)
    return ans_v * np.linalg.norm(v) / np.linalg.norm(ans_v) #微調整

class Point:
    def __init__(self, abs_coordinate, color, radius):
        self.abs_coordinate = abs_coordinate
        self.color = color
        self.radius = radius

    def calc_distance(self, point2):
        return np.linalg.norm(self.abs_coordinate - point2.abs_coordinate)

    def get_radius(self):
        return self.radius

    def get_abs_coordinate(self):
        return self.abs_coordinate

    def get_color(self):
        return self.color


class Line:
    def __init__(self, point1:Point, point2:Point, color, width):
        self.point1 = point1
        self.point2 = point2
        self.color = color
        self.width = width

    def get_point1(self):
        return self.point1

    def get_point2(self):
        return self.point2

    def get_color(self):
        return self.color

    def get_width(self):
        return self.width

    def get_center_abs_coordinate(self):
        return (self.point1.abs_coordinate + self.point2.abs_coordinate) / 2

class Polygon2d:
    def __init__(self, point1, point2, point3, color):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.color = color

    def get_points(self):
        return self.point1, self.point2, self.point3

    def get_color(self):
        return self.color

    def get_center(self):
        return (self.point1.get_abs_coordinate + self.point2.get_abs_coordinate + self.point3.get_abs_coordinate) / 3


class Polygon3d:
    def __init__(self, point1, point2, point3, point4, color):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.color = color

    def get_points(self):
        return self.point1, self.point2, self.point3, self.point4

    def get_color(self):
        return self.color

    def get_center(self):
        return (self.point1.get_abs_coordinate + self.point2.get_abs_coordinate + self.point3.get_abs_coordinate + self.point4.get_abs_coordinate) / 4


class Camera:
    def __init__(self, first_abs_coordinate, camera_coordinate_axises, aspect_ratio, max_visibility, FOV, max_alpha, screen_distance):
        self.abs_coordinate = first_abs_coordinate
        self.camera_coordinate_axises = {
            "w": camera_coordinate_axises[0],
            "x": camera_coordinate_axises[1],
            "y": camera_coordinate_axises[2],
            "z": camera_coordinate_axises[3],
        }
        self.screen_size = np.array((1, aspect_ratio[1]/ aspect_ratio[0]))
        self.max_visibility = max_visibility
        self.FOV = FOV
        self.max_alpha = max_alpha
        self.screen_distance = screen_distance

    def rotate_with_PlaneWX(self, theta):
        rotated_y = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["x"]),self.camera_coordinate_axises["y"],theta)
        rotated_z = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["x"]),self.camera_coordinate_axises["z"],theta)
        self.camera_coordinate_axises["y"] = rotated_y
        self.camera_coordinate_axises["z"] = rotated_z

    def rotate_with_PlaneWY(self, theta):
        rotated_x = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["y"]),self.camera_coordinate_axises["x"],theta)
        rotated_z = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["y"]),self.camera_coordinate_axises["z"],theta)
        self.camera_coordinate_axises["x"] = rotated_x
        self.camera_coordinate_axises["z"] = rotated_z

    def rotate_with_PlaneWZ(self, theta):
        rotated_x = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["x"],theta)
        rotated_y = rotate_with_planeP((self.camera_coordinate_axises["w"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["y"],theta)
        self.camera_coordinate_axises["x"] = rotated_x
        self.camera_coordinate_axises["y"] = rotated_y

    def rotate_with_PlaneXY(self, theta):
        rotated_w = rotate_with_planeP((self.camera_coordinate_axises["x"],self.camera_coordinate_axises["y"]),self.camera_coordinate_axises["w"],theta)
        rotated_z = rotate_with_planeP((self.camera_coordinate_axises["x"],self.camera_coordinate_axises["y"]),self.camera_coordinate_axises["z"],theta)
        self.camera_coordinate_axises["w"] = rotated_w
        self.camera_coordinate_axises["z"] = rotated_z

    def rotate_with_PlaneXZ(self, theta):
        rotated_w = rotate_with_planeP((self.camera_coordinate_axises["x"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["w"],theta)
        rotated_y = rotate_with_planeP((self.camera_coordinate_axises["x"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["y"],theta)
        self.camera_coordinate_axises["w"] = rotated_w
        self.camera_coordinate_axises["y"] = rotated_y

    def rotate_with_PlaneYZ(self, theta):
        rotated_w = rotate_with_planeP((self.camera_coordinate_axises["y"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["w"],theta)
        rotated_x = rotate_with_planeP((self.camera_coordinate_axises["y"],self.camera_coordinate_axises["z"]),self.camera_coordinate_axises["x"],theta)
        self.camera_coordinate_axises["w"] = rotated_w
        self.camera_coordinate_axises["x"] = rotated_x

    def move_to_x(self, rate=1.0):
        self.abs_coordinate += self.camera_coordinate_axises["x"] * rate

    def move_to_y(self, rate=1.0):
        self.abs_coordinate += self.camera_coordinate_axises["y"] * rate

    def move_to_z(self, rate=1.0):
        self.abs_coordinate += self.camera_coordinate_axises["z"] * rate

    def move_to_w(self, rate=1.0):
        self.abs_coordinate += self.camera_coordinate_axises["w"] * rate

    def reset_coordinate_and_rotation(self):
        self.abs_coordinate = np.array((0.0, 0.0, 0.0, 0.0))
        self.camera_coordinate_axises = {
            "w": np.array((1.0, 0.0, 0.0, 0.0)),
            "x": np.array((0.0, 1.0, 0.0, 0.0)),
            "y": np.array((0.0, 0.0, 1.0, 0.0)),
            "z": np.array((0.0, 0.0, 0.0, 1.0)),
        }

    def is_point_in_vision(self, point_camera_coordinate): #カメラ前方(視野角内)
        return np.dot(point_camera_coordinate, self.camera_coordinate_axises["z"]) / np.linalg.norm(point_camera_coordinate) > np.cos(self.FOV / 2)

    def draw_point_sets(self, point_sets):
        for point_set in point_sets:
            for point in point_set:
                if self.is_point_in_vision(point.abs_coordinate):
