from voxel import read_vox_file
import numpy as np

class Texture:
    def __init__(self, voxel_path: str):
        self.voxels = read_vox_file(voxel_path)

class Block:
    def __init__(self,texture:Texture, x:int, y:int):
        """
        X ∈ [x,x+1], Y ∈ [y,y+1], Z ∈ [z,z+1]
        :param texture:
        :param x:
        :param y:
        """
        self.x = x
        self.y = y
        self.texture = texture

def calc_vision(camera_point, camera_vector, points):
    vectors_AP = points - camera_point
    z = camera_vector / np.linalg.norm(camera_vector)
    if np.array_equal(z, np.array((1,0,0))) or np.array_equal(z, np.array((-1,0,0))):
        x = np.cross(z , (0,1,0))
    else:
        x = np.cross(z , (1,0,0))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    relative_coordinates_tensor = np.array((x,y,z)).T
    relative_points = np.linalg.solve(relative_coordinates_tensor, vectors_AP.T)

    return relative_points.reshape((1,3))