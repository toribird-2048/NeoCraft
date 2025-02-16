import pygame
import numpy as np
import quaternion
import sys
import pytest

#視界に入る立体範囲（視界立体）:view_solid


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

    def calc_alpha_from_distance(self, distance): # alpha∈[0,1]
        return max((self.max_visibility - distance) / self.max_visibility, 0) * self.max_alpha

    def camera_coordinate_to_screen_coordinate(self, point_camera_coordinate):
        point_screen_coordinate_x = np.dot(point_camera_coordinate,self.camera_coordinate_axises["x"]) * self.screen_distance / np.dot(point_camera_coordinate, self.camera_coordinate_axises["z"])
        point_screen_coordinate_x = point_screen_coordinate_x * 2 / self.screen_size[0]
        point_screen_coordinate_y = np.dot(point_camera_coordinate,self.camera_coordinate_axises["y"]) * self.screen_distance / np.dot(point_camera_coordinate, self.camera_coordinate_axises["z"])
        point_screen_coordinate_y = point_screen_coordinate_y * 2 / self.screen_size[0]
        point_screen_coordinate = np.array((point_screen_coordinate_x, point_screen_coordinate_y))
        print(f"in_func_CCSC: {np.dot(point_camera_coordinate, self.camera_coordinate_axises["z"])}")
        return point_screen_coordinate

    def is_point_in_vision(self, point_camera_coordinate): #カメラ前方(視野角内)
        return np.dot(point_camera_coordinate, self.camera_coordinate_axises["z"]) / np.linalg.norm(point_camera_coordinate) > np.cos(self.FOV / 2)

    def is_in_view_solid(self, point_camera_coordinate):
        tolerance = 1e-6
        return abs(np.dot(point_camera_coordinate, self.camera_coordinate_axises["w"])) < tolerance

    def draw_point_sets_screen_coordinate(self,screen, pygame_screen_size, point_sets:list[list[Point]]):
        point_screen_coordinates = []
        point_radiuses = []
        point_colors = []
        for point_set in point_sets:
            for point in point_set:
                point_camera_coordinate = point.get_abs_coordinate() - self.abs_coordinate
                if self.is_point_in_vision(point_camera_coordinate):
                    if self.is_in_view_solid(point_camera_coordinate): #FractalContinueNote1_α_参照
                        point_screen_coordinate = self.camera_coordinate_to_screen_coordinate(point_camera_coordinate)
                        point_radius = point.get_radius()
                        alpha = self.calc_alpha_from_distance(point.calc_distance(self))
                        color = np.append(point.color,alpha)
                        point_screen_coordinates.append(point_screen_coordinate)
                        point_radiuses.append(point_radius)
                        point_colors.append(color)
        draw_points(screen=screen, pygame_screen_size=pygame_screen_size, point_screen_coordinates=point_screen_coordinates, point_radiuses=point_radiuses, colors=point_colors)


    def draw_line_sets_screen_coordinate(self, screen, pygame_screen_size, line_sets:list[list[Line]]):
        line_point1_screen_coordinates = []
        line_point2_screen_coordinates = []
        line_widthes = []
        line_colors = []

        points = []
        point_abs_coordinates = []
        point_radiuses = []
        point_colors = []
        for line_set in line_sets:
            for line in line_set:
                point1_abs_coordinate = line.get_point1().get_abs_coordinate()
                point2_abs_coordinate = line.get_point2().get_abs_coordinate()
                point1_camera_coordinate = point1_abs_coordinate - self.abs_coordinate
                point2_camera_coordinate = point2_abs_coordinate - self.abs_coordinate
                camera_coordinate_axises = self.camera_coordinate_axises

                if self.is_point_in_vision(point1_camera_coordinate) and self.is_point_in_vision(point2_camera_coordinate): #線の始点と終点が両方視線ベクトル方向にある
                    if self.is_in_view_solid(point1_camera_coordinate) and self.is_in_view_solid(point2_camera_coordinate): #線の視点と終点が視界立体の内側にある
                        print(point1_camera_coordinate, point2_camera_coordinate)
                        start_point_screen_coordinate = self.camera_coordinate_to_screen_coordinate(point1_camera_coordinate)
                        end_point_screen_coordinate = self.camera_coordinate_to_screen_coordinate(point2_camera_coordinate)
                        start_point_abs_coordinate = point1_abs_coordinate
                        end_point_abs_coordinate = point2_abs_coordinate
                        line_point1_screen_coordinates.append(start_point_screen_coordinate)
                        line_point2_screen_coordinates.append(end_point_screen_coordinate)
                        line_widthes.append(line.width)
                        line_colors.append(line.color)
                        point_abs_coordinates.append(start_point_abs_coordinate)
                        point_abs_coordinates.append(end_point_abs_coordinate)
                        point_radiuses.append(line.width)
                        point_radiuses.append(line.width)
                        point_colors.append(line.color)
                        point_colors.append(line.color)
                        print(1)
                    else:
                        t = -np.dot(point2_camera_coordinate, camera_coordinate_axises["w"]) / np.dot(point1_camera_coordinate - point2_camera_coordinate, camera_coordinate_axises["w"])
                        if 0 <= t <= 1: #線のどこかの点が視界立体の内側にある
                            point_camera_coordinate = t * point1_camera_coordinate + (1 - t) * point2_camera_coordinate
                            point_abs_coordinate = point_camera_coordinate + self.abs_coordinate
                            point_abs_coordinates.append(point_abs_coordinate)
                            point_radiuses.append(line.width)
                            point_colors.append(line.color)
                            print(2)
                        else:
                            #線のどの点も視界立体の内側にない
                            pass
                elif self.is_point_in_vision(point1_camera_coordinate) and not self.is_point_in_vision(point2_camera_coordinate): #始点だけがカメラ前方にある（終点は後方）
                    s = (np.cos(self.FOV / 2) - np.dot(point2_camera_coordinate, camera_coordinate_axises["z"])) / np.dot(point1_camera_coordinate - point2_camera_coordinate, camera_coordinate_axises["z"])
                    if 0 <= s <= 1:
                        if self.is_in_view_solid(point1_camera_coordinate) and self.is_in_view_solid(point2_camera_coordinate): #線の視点と終点が視界立体の内側にある
                            start_point_screen_coordinate = self.camera_coordinate_to_screen_coordinate(point1_camera_coordinate)
                            point_camera_coordinate = s * point1_camera_coordinate + (1 - s) * point2_camera_coordinate
                            point_screen_coordinate = self.camera_coordinate_to_screen_coordinate(point_camera_coordinate)
                            start_point_abs_coordinate = point1_abs_coordinate
                            point_abs_coordinate = point_camera_coordinate + self.abs_coordinate
                            line_point1_screen_coordinates.append(start_point_screen_coordinate)
                            line_point2_screen_coordinates.append(point_screen_coordinate)
                            line_widthes.append(line.width)
                            line_colors.append(line.color)
                            point_abs_coordinates.append(start_point_abs_coordinate)
                            point_abs_coordinates.append(point_abs_coordinate)
                            point_radiuses.append(line.width)
                            point_radiuses.append(line.width)
                            point_colors.append(line.color)
                            point_colors.append(line.color)
                            print(3)
                        elif not (self.is_in_view_solid(point1_camera_coordinate) or self.is_in_view_solid(point1_camera_coordinate)): #線の始点と終点の両方が視界立体の内側にない
                            t = -np.dot(point2_camera_coordinate, camera_coordinate_axises["w"]) / np.dot(point1_camera_coordinate - point2_camera_coordinate, camera_coordinate_axises["w"])
                            if 0 <= t <= 1:
                                point_camera_coordinate = t * point1_camera_coordinate + (1 - t) * point2_camera_coordinate
                                if self.is_point_in_vision(point_camera_coordinate): #線のうちの、w=0の点がカメラ前方にある
                                    point_abs_coordinate = point_camera_coordinate + self.abs_coordinate
                                    point_abs_coordinates.append(point_abs_coordinate)
                                    point_radiuses.append(line.width)
                                    point_colors.append(line.color)
                                    print(4)
                                else:
                                    #線のうちの、w=0の点がカメラ前方にない
                                    pass
                            else:
                                #線のどの点も視界立体の内側にない
                                pass
                        elif self.is_in_view_solid(point1_camera_coordinate):#線の始点だけが視界立体の内側にある
                            if self.is_point_in_vision(point1_camera_coordinate):#線の始点がカメラ前方にある
                                point_abs_coordinate = point1_abs_coordinate
                                point_abs_coordinates.append(point_abs_coordinate)
                                point_radiuses.append(line.width)
                                point_colors.append(line.color)
                                print(5)
                            else:
                                #線の始点がカメラ前方にない
                                pass
                        elif self.is_in_view_solid(point2_camera_coordinate):#線の終点だけが視界立体の内側にある
                            if self.is_point_in_vision(point2_camera_coordinate):#線の終点がカメラ前方にある
                                point_abs_coordinate = point2_abs_coordinate
                                point_abs_coordinates.append(point_abs_coordinate)
                                point_radiuses.append(line.width)
                                point_colors.append(line.color)
                                print(6)
                            else:
                                #線の終点がカメラ前方にないい
                                pass

        for point_screen_coordinate, point_radius, color in zip(point_abs_coordinates, point_radiuses, point_colors):
            points.append(Point(abs_coordinate=point_screen_coordinate, color=color, radius=point_radius))
        self.draw_point_sets_screen_coordinate(screen=screen, pygame_screen_size=pygame_screen_size, point_sets=[points])
        draw_lines(screen=screen, pygame_screen_size=pygame_screen_size, point1_screen_coordinates=line_point1_screen_coordinates, point2_screen_coordinates=line_point2_screen_coordinates, line_widthes=line_widthes, colors=line_colors)







def draw_points(screen, pygame_screen_size, point_screen_coordinates, point_radiuses, colors):
    surface = pygame.Surface(pygame_screen_size, pygame.SRCALPHA)
    for point_screen_coordinate, point_radius, color in zip(point_screen_coordinates, point_radiuses, colors):
        point_pygame_color = np.array(255 * color, dtype=np.uint8)
        point_pygame_color = pygame.Color(list(point_pygame_color))
        pygame_screen_width = pygame_screen_size[0]
        point_pygame_coordinate = pygame_screen_width * point_screen_coordinate / 2 + (pygame_screen_size[0] / 2, -pygame_screen_size[1] / 2)
        point_pygame_coordinate[1] = -point_pygame_coordinate[1]
        pygame.draw.circle(surface, point_pygame_color, point_pygame_coordinate, point_radius)
    screen.blit(surface, (0, 0))

def draw_lines(screen, pygame_screen_size, point1_screen_coordinates, point2_screen_coordinates, line_widthes, colors):
    surface = pygame.Surface(pygame_screen_size, pygame.SRCALPHA)
    for point1_screen_coordinate, point2_screen_coordinate, line_width, color in zip(point1_screen_coordinates,point2_screen_coordinates, line_widthes, colors):
        line_pygame_color = np.array(255 * color, dtype=np.uint8)
        line_pygame_color = pygame.Color(list(line_pygame_color))
        pygame_screen_width = pygame_screen_size[0]
        point1_pygame_coordinate = pygame_screen_width * point1_screen_coordinate / 2 + (pygame_screen_size[0] / 2, -pygame_screen_size[1] / 2)
        point1_pygame_coordinate[1] = -point1_pygame_coordinate[1]
        point2_pygame_coordinate = pygame_screen_width * point2_screen_coordinate / 2 + (pygame_screen_size[0] / 2, -pygame_screen_size[1] / 2)
        point2_pygame_coordinate[1] = -point2_pygame_coordinate[1]
        pygame.draw.line(surface, line_pygame_color, point1_pygame_coordinate, point2_pygame_coordinate, width=line_width)
    screen.blit(surface, (0,0))

def generate_points(n, radius=5):
    return [Point(np.array((0, 2 * radius * rng.random() - radius, 2 * radius * rng.random() - radius, 2 * radius * rng.random() - radius)), rng.random(3), 5) for _ in range(n)]

def generate_lines(n, width=5):
    point1_set = generate_points(n)
    point2_set = generate_points(n)
    ans = []
    for k in range(n):
        ans.append(Line(point1_set[k], point2_set[k], rng.random(3), width))
    return ans


point_sets = [
    [
        Point(np.array((0.0, 2.0, 2.0, 2.0)), np.array((0,0,1)), 10),
        Point(np.array((0.0, -2.0, 2.0, 2.0)), np.array((0,1,0)), 10),
        Point(np.array((0.0, 2.0, -2.0, 2.0)), np.array((0,1,1)), 10),
        Point(np.array((0.0, -2.0, -2.0, 2.0)), np.array((1,0,0)), 10),
        Point(np.array((0.0, 2.0, 2.0, -2.0)), np.array((1,0,1)), 10),
        Point(np.array((0.0, -2.0, 2.0, -2.0)), np.array((1,1,0)), 10),
        Point(np.array((0.0, 2.0, -2.0, -2.0)), np.array((0.5,0.5,1)), 10),
        Point(np.array((0.0, -2.0, -2.0, -2.0)), np.array((1,0.5,0.5)), 10),
    ],
    #[Point(np.array((0.0,10*rng.random()-5,10*rng.random()-5,10*rng.random()-5)), rng.random(3), 5) for _ in range(100)]
]

line_sets = [
    [
        #Line(Point(np.array((0,0,0,1)),np.array((1,1,1)),5),Point(np.array((0,0,1,1)),np.array((1,1,1)),5),np.array((1,1,1)),5),
        Line(Point(np.array((0.0,0.0,0.0,1.0)),np.array((1,1,1)),5),Point(np.array((0.0,1.0,0.0,0.0)),np.array((1,1,1)),5),np.array((1,1,0)),5)
    ],
    #generate_lines(20)
]

screen_aspect_ratio = np.array((1,1))

main_camera = Camera(np.array((0.0, 0.0, 0.0, 0.0)), [np.array((1.0, 0.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0, 0.0)), np.array((0.0, 0.0, 0.0, 1.0))], screen_aspect_ratio, 20, 160, 0.8,  1)

rotate_speed = 0.02
move_speed = 0.07

PYGAME_SCREEN_WIDTH = 800

pygame_screen_size = (PYGAME_SCREEN_WIDTH, PYGAME_SCREEN_WIDTH * screen_aspect_ratio[1] / screen_aspect_ratio[0]) #関数では横幅を基準とし、1とおく。

PYGAME_BACKGROUND_COLOR = (40,40,40)

pygame.init()

screen = pygame.display.set_mode(pygame_screen_size, pygame.SRCALPHA)

pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                main_camera.reset_coordinate_and_rotation()

        key_get_pressed = pygame.key.get_pressed()
        #回転
        if key_get_pressed[pygame.K_f]:
            main_camera.rotate_with_PlaneWX(rotate_speed)
        if key_get_pressed[pygame.K_r]:
            main_camera.rotate_with_PlaneWX(-rotate_speed)
        if key_get_pressed[pygame.K_e]:
            main_camera.rotate_with_PlaneWY(rotate_speed)
        if key_get_pressed[pygame.K_q]:
            main_camera.rotate_with_PlaneWY(-rotate_speed)
        if key_get_pressed[pygame.K_o]:
            main_camera.rotate_with_PlaneXY(rotate_speed)
        if key_get_pressed[pygame.K_u]:
            main_camera.rotate_with_PlaneXY(-rotate_speed)
        #移動
        if key_get_pressed[pygame.K_l]:
            main_camera.move_to_w(move_speed)
        if key_get_pressed[pygame.K_j]:
            main_camera.move_to_w(-move_speed)
        if key_get_pressed[pygame.K_d]:
            main_camera.move_to_x(move_speed)
        if key_get_pressed[pygame.K_a]:
            main_camera.move_to_x(-move_speed)
        if key_get_pressed[pygame.K_i]:
            main_camera.move_to_y(move_speed)
        if key_get_pressed[pygame.K_k]:
            main_camera.move_to_y(-move_speed)
        if key_get_pressed[pygame.K_w]:
            main_camera.move_to_z(move_speed)
        if key_get_pressed[pygame.K_s]:
            main_camera.move_to_z(-move_speed)

    screen.fill(PYGAME_BACKGROUND_COLOR)

    #draw_point(screen=screen, pygame_screen_size=PYGAME_SCREEN_SIZE, point_screen_coordinate=np.array((0,0)), point_radius=10, color=np.array((1,1,1,0.5)))
    #draw_line(screen=screen, pygame_screen_size=PYGAME_SCREEN_SIZE, point1_screen_coordinate=np.array((-1,-1)), point2_screen_coordinate=np.array((1,1)), line_width=3, color=np.array((1,1,1,0.5)))
    main_camera.draw_point_sets_screen_coordinate(screen=screen, pygame_screen_size=pygame_screen_size, point_sets=point_sets)
    main_camera.draw_line_sets_screen_coordinate(screen=screen, pygame_screen_size=pygame_screen_size, line_sets=line_sets)

    pygame.display.flip()

    pygame.time.Clock().tick(60)