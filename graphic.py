import sys

import pygame
import numpy as np
import quaternion
import time


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
    return np.array([ans_q_v.w, ans_q_v.x, ans_q_v.y, ans_q_v.z])


class Camera:
    def __init__(self, abs_position, w, x, y, z, screen3d_distance = 1.0):
        self.abs_position = np.array(abs_position, dtype=np.float64)
        self.w = np.array(w, dtype=np.float64)  # float64型で初期化
        self.x = np.array(x, dtype=np.float64)  # float64型で初期化
        self.y = np.array(y, dtype=np.float64)  # float64型で初期化
        self.z = np.array(z, dtype=np.float64)  # float64型で初期化
        self.screen3d_distance = screen3d_distance

    def rotate_with_planeWX(self, theta):
        rotated_y = rotate_with_planeP((self.w,self.x),self.y,theta)
        rotated_z = rotate_with_planeP((self.w,self.x),self.z,theta)
        self.y = rotated_y
        self.z = rotated_z

    def rotate_with_planeWY(self, theta):
        rotated_x = rotate_with_planeP((self.w,self.y),self.x,theta)
        rotated_z = rotate_with_planeP((self.w,self.y),self.z,theta)
        self.x = rotated_x
        self.z = rotated_z

    def rotate_with_planeWZ(self, theta):
        rotated_x = rotate_with_planeP((self.w,self.z),self.x,theta)
        rotated_y = rotate_with_planeP((self.w,self.z),self.y,theta)
        self.x = rotated_x
        self.y = rotated_y


    def rotate_with_planeXY(self, theta):
        rotated_w = rotate_with_planeP((self.x,self.y),self.w,theta)
        rotated_z = rotate_with_planeP((self.x,self.y),self.z,theta)
        self.w = rotated_w
        self.z = rotated_z

    def rotate_with_planeXZ(self, theta):
        rotated_w = rotate_with_planeP((self.x,self.z),self.w,theta)
        rotated_y = rotate_with_planeP((self.x,self.z),self.y,theta)
        self.w = rotated_w
        self.y = rotated_y

    def rotate_with_planeYZ(self, theta):
        rotated_w = rotate_with_planeP((self.y,self.z),self.w,theta)
        rotated_x = rotate_with_planeP((self.y,self.z),self.x,theta)
        self.w = rotated_w
        self.x = rotated_x

    def move_to_x(self, rate=1.0):
        self.abs_position += self.x * rate

    def move_to_y(self, rate=1.0):
        self.abs_position += self.y * rate

    def move_to_z(self, rate=1.0):
        self.abs_position += self.z * rate

    def move_to_w(self, rate=1.0):
        self.abs_position += self.w * rate

class Point:
    def __init__(self, abs_position, camera:Camera, color = (255, 255, 255), radius = 5):
        self.abs_position = abs_position
        self.color = color
        self.camera = camera
        self.screen3d_position = (0,0,0)
        self.screen2d_position = (0,0)
        self.radius = radius

    def calc_screen3d_point(self):
        vector_vp = self.abs_position - self.camera.abs_position
        vector_w = self.camera.w
        vector_x = self.camera.x
        vector_y = self.camera.y
        vector_z = self.camera.z
        d = self.camera.screen3d_distance
        z_vp_vector = np.dot(vector_z, vector_vp) * vector_z
        w_vp_vector = np.dot(vector_w, vector_vp) * vector_w
        x_vp_vector = np.dot(vector_x, vector_vp) * vector_x
        y_vp_vector = np.dot(vector_y, vector_vp) * vector_y
        z_norm = np.linalg.norm(z_vp_vector)
        if z_norm < 1e-6:  # 1e-6 は非常に小さい値（epsilon）の例。適宜調整してください。
            self.screen3d_position = (None, None, None)  # z_norm がほぼゼロの場合は None を設定
        else:
            a = np.linalg.norm(w_vp_vector) * d / z_norm
            b = np.linalg.norm(x_vp_vector) * d / z_norm
            c = np.linalg.norm(y_vp_vector) * d / z_norm
            self.screen3d_position = a, b, c

    def calc_screen2d_point(self, position_3d_camera=np.array((0,0,0)), vector_x_3d_camera=np.array((1,0,0)), vector_y_3d_camera=np.array((0,1,0)), vector_z_3d_camera=np.array((0,0,1)), d=1):
        if self.screen3d_position == (None, None, None): # screen3d_position が None の場合は計算をスキップ
            self.screen2d_position = (None, None)
            return

        vector_vp_3d_camera = self.screen3d_position - position_3d_camera
        z_vp_vector = np.dot(vector_z_3d_camera, vector_vp_3d_camera) * vector_z_3d_camera
        x_vp_vector = np.dot(vector_x_3d_camera, vector_vp_3d_camera) * vector_x_3d_camera
        y_vp_vector = np.dot(vector_y_3d_camera, vector_vp_3d_camera) * vector_y_3d_camera
        z_norm = np.linalg.norm(z_vp_vector)
        if z_norm < 1e-6: # 1e-6 は非常に小さい値（epsilon）の例。適宜調整してください。
            self.screen2d_position = (None, None) # z_norm がほぼゼロの場合は None を設定
        else:
            a = np.linalg.norm(x_vp_vector) * d / z_norm
            b = np.linalg.norm(y_vp_vector) * d / z_norm
            self.screen2d_position = a, b

    def calc_4d_camera_distance(self):
        return np.linalg.norm(self.abs_position - self.camera.abs_position)


def generate_random_points(n, camera_coordinate, size=5):
    return [Point(rng.random(4), camera_coordinate, (rng.random(3)*255).astype(int), size) for _ in range(n)]

def calc_color_from_distance(point, max_distance):
    point_color = point.color
    distance = point.calc_4d_camera_distance()
    return [int(color_arg) for color_arg in point_color * max((max_distance - distance) / max_distance, 0)]

def calc_point_game_screen(zoom, point, screen_size):
    if point == (None, None) or np.isnan(point[0]) or np.isnan(point[1]): # point が None または NaN を含む場合は None を返す
        return None
    return int(zoom * point[0] - screen_size[0] / 2), int(zoom * point[1] - screen_size[1] / 2)



camera_coordinate = Camera(np.array((0, 0, 0, 0)), np.array((1, 0, 0, 0)), np.array((0, 1, 0, 0)), np.array((0, 0, 1, 0)), np.array((0, 0, 0, 1)))

point_sets = [
    [
        Point(np.array((0.01, 0, 0, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((-0.01, 0, 0, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, 0.01, 0, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, -0.01, 0, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, 0, 0.01, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, 0, -0.01, 0)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, 0, 0, 0.01)), camera_coordinate, np.array((255, 255, 255)), 10),
        Point(np.array((0, 0, 0, -0.01)), camera_coordinate, np.array((255, 255, 255)), 10),
    ],
    generate_random_points(1000, camera_coordinate)
]

screen3d_distance = 1

def render_to_3d_screen(point_sets,camera_coordinate):
    camera_position = camera_coordinate.abs_position
    camera_w = camera_coordinate.w
    camera_x = camera_coordinate.x
    camera_y = camera_coordinate.y
    camera_z = camera_coordinate.z

    for k, point_set in enumerate(point_sets):
        for l, point in enumerate(point_set):
            point.calc_screen3d_point()

def render_to_2d_screen(point_sets_screen3d):

    for k, point_set in enumerate(point_sets_screen3d):
        for l, point in enumerate(point_set):
            point.calc_screen2d_point()



SCREEN_SIZE = (800, 800)
move_rate = 0.001
max_distance = 1.5
zoom_rate = 100


pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()

pygame.key.set_repeat(500, 10)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_w:
                camera_coordinate.move_to_z(move_rate)
            elif event.key == pygame.K_s:
                camera_coordinate.move_to_z(-move_rate)
            elif event.key == pygame.K_a:
                camera_coordinate.move_to_x(-move_rate)
            elif event.key == pygame.K_d:
                camera_coordinate.move_to_x(move_rate)
            elif event.key == pygame.K_i:
                camera_coordinate.move_to_y(move_rate)
            elif event.key == pygame.K_k:
                camera_coordinate.move_to_y(-move_rate)
            elif event.key == pygame.K_j:
                camera_coordinate.move_to_w(-move_rate)
            elif event.key == pygame.K_l:
                camera_coordinate.move_to_w(move_rate)
            elif event.key == pygame.K_q:
                camera_coordinate.rotate_with_planeXY(move_rate)
            elif event.key == pygame.K_e:
                camera_coordinate.rotate_with_planeXY(-move_rate)
            elif event.key == pygame.K_r:
                camera_coordinate.rotate_with_planeWY(move_rate)
            elif event.key == pygame.K_f:
                camera_coordinate.rotate_with_planeWY(-move_rate)
            elif event.key == pygame.K_u:
                camera_coordinate.rotate_with_planeWX(move_rate)
            elif event.key == pygame.K_o:
                camera_coordinate.rotate_with_planeWX(-move_rate)


            print(camera_coordinate.w, camera_coordinate.x, camera_coordinate.y, camera_coordinate.z)


    screen.fill((0, 0, 0))

    render_to_3d_screen(point_sets, camera_coordinate)
    render_to_2d_screen(point_sets)
    for k, point_set in enumerate(point_sets):
        for l, point in enumerate(point_set):
            color = calc_color_from_distance(point, max_distance)
            point_2d_screen = point.screen2d_position
            point_game_screen = calc_point_game_screen(zoom_rate, point_2d_screen, SCREEN_SIZE)
            if point_game_screen is not None:  # point_game_screen が None でない場合のみ描画
                pygame.draw.circle(screen, color, point_game_screen, point.radius)

    pygame.display.flip()
    clock.tick(60)
