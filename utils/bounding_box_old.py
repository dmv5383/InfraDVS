import numpy as np
import carla

from typing import Any, Dict, List, Tuple, Callable

def get_global_bounding_boxes(actors: Any) -> List:
    bounding_boxes = []
    for actor in actors:
        bounding_box = actor.bounding_box
        bounding_boxes.append(bounding_box)

    return bounding_boxes

def bounding_boxes_world_to_sensor(bounding_boxes: List, sensor: Any, is_lidar: bool) -> np.ndarray:
    sensor_bp = sensor.get_bp()
    sensor_obj = sensor.get_obj()
    world_to_sensor_matrix = np.array(sensor_obj.get_transform().get_inverse_matrix())

    output_vertices = []

    range = 120
    if sensor_bp.has_attribute("range"):
        range = sensor_bp.get_attribute("range").as_float()

    for bb in bounding_boxes:
        dist = bb.location.distance(sensor_obj.get_transform().location)

        if dist <= range:
            # Get 2D bounding boxes in front on sensor
            if not is_lidar:
                image_w = sensor_bp.get_attribute("image_size_x").as_int()
                image_h = sensor_bp.get_attribute("image_size_y").as_int()
                fov = sensor_bp.get_attribute("fov").as_float()
                K = build_projection_matrix(image_w, image_h, fov)

                forward_vec = sensor_obj.get_transform().get_forward_vector()
                ray = bb.location - sensor_obj.get_transform().location

                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = get_image_point(vert, K, world_to_sensor_matrix)
                        # Find the rightmost vertex
                        if p[0] > x_max:
                            x_max = p[0]
                        # Find the leftmost vertex
                        if p[0] < x_min:
                            x_min = p[0]
                        # Find the highest vertex
                        if p[1] > y_max:
                            y_max = p[1]
                        # Find the lowest  vertex
                        if p[1] < y_min:
                            y_min = p[1]

                    width = x_max - x_min
                    height = y_max - y_min
                    output_vertices.append([x_min, y_min, width, height])

            # Get 3D bounding boxes in all directions
            else:
                sensor_verts = []
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                for vert in verts:
                    point = np.array([vert.x, vert.y, vert.z, 1])
                    transformed_point = np.dot(world_to_sensor_matrix, point)
                    sensor_verts.append(transformed_point[:3])  # Only take x, y, z

                output_vertices.append(np.array(sensor_verts).flatten())
    if not output_vertices and not is_lidar:
        return np.empty((0, 4))
    return np.array(output_vertices)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]