from typing import Any, List
import carla
import numpy as np

from utils.datadescriptor import KittiDescriptor
from utils.camera import *

import math
import logging

OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)
MIN_VISIBLE_VERTICES_FOR_RENDER = 2
MIN_BBOX_AREA_IN_PX = 100  # Adjust as required.


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera, h, w, fov):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [(ClientSideBoundingBoxes.get_bounding_box(
            vehicle, camera, h, w, fov), vehicle) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[0][:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box(vehicle, camera, h, w, fov):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, camera)[:3, :]

        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])

        calibration = np.identity(3)
        calibration[0, 2] = w / 2.0
        calibration[1, 2] = h / 2.0
        calibration[0, 0] = calibration[1, 1] = w / \
            (2.0 * np.tan(fov * np.pi / 360.0))

        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def get_bounding_box_lidar(vehicle, lidar_sensor):
        """
        Returns 3D bounding box for a vehicle based on LiDAR sensor data.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, lidar_sensor)[:3, :]

        return cords_x_y_z

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(
            world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(
            vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(
            sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    @staticmethod
    def get_bounding_boxes_parked_vehicles(bboxes, camera, h, w, fov):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box_parked_vehicle(
            vehicle, camera, h, w, fov) for vehicle in bboxes]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box_parked_vehicle(bbox, camera, h, w, fov):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = ClientSideBoundingBoxes._bounding_box_to_world(bbox)
        cords_x_y_z = ClientSideBoundingBoxes._world_to_sensor(bb_cords, camera)[
            :3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        calibration = np.identity(3)
        calibration[0, 2] = w / 2.0
        calibration[1, 2] = h / 2.0
        calibration[0, 0] = calibration[1, 1] = w / \
            (2.0 * np.tan(fov * np.pi / 360.0))
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _bounding_box_to_world(bbox):
        extent = bbox.extent
        cords = np.zeros((8, 4))
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

        world_matrix = ClientSideBoundingBoxes.get_matrix(bbox)

        world_cords = np.dot(world_matrix, np.transpose(cords))

        return world_cords

    @staticmethod
    def _create_bb_points_parked(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        if isinstance(vehicle, carla.BoundingBox):
            extent = vehicle.extent
        else:
            extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

def get_bboxes_2d(sensor: Any, depth: np.ndarray, world: Any) -> List:
    vehicles = world.get_actors().filter("vehicle.*")
    pedestrians = world.get_actors().filter("walker.*")

    bounding_boxes = []

    vehicle_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(
        vehicles=vehicles,
        camera=sensor,
        h=int(sensor.attributes.get("image_size_y")),
        w=int(sensor.attributes.get("image_size_x")),
        fov=float(sensor.attributes.get("fov"))
    )
    for bbox in vehicle_bboxes:
        points = [(int(bbox[0][i, 0]), int(bbox[0][i, 1])) for i in range(8)]
        min_x, min_y, xdiff, ydiff = get_2d_bounding_box(
            np.array(points, dtype=np.int32)
        )
        vehicle = bbox[1]
        
        # Check for occlusion
        occlusion_status = check_occlusion(points, bbox[0][:, 2], depth, min_x, min_y, xdiff, ydiff)
        
        if occlusion_status < 2:
            # Only add bounding box if it's not largely occluded
            bounding_boxes.append((vehicle.id, vehicle.attributes.get("base_type"),
                                (min_x, min_y, xdiff, ydiff), bbox[0]))
            #print(f"Vehicle {vehicle.id} is visible, adding bounding box.")
        #else:
            #print(f"Vehicle {vehicle.id} is largely occluded, skipping bounding box.")
        
    ped_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(
        vehicles=pedestrians,
        camera=sensor,
        h=int(sensor.attributes.get("image_size_y")),
        w=int(sensor.attributes.get("image_size_x")),
        fov=float(sensor.attributes.get("fov"))
    )
    for bbox in ped_bboxes:
        points = [(int(bbox[0][i, 0]), int(bbox[0][i, 1])) for i in range(8)]
        min_x, min_y, xdiff, ydiff = get_2d_bounding_box(
            np.array(points, dtype=np.int32)
        )
        ped = bbox[1]
        
        # Check for occlusion
        occlusion_status = check_occlusion(points, bbox[0][:, 2], depth, min_x, min_y, xdiff, ydiff)
        
        if occlusion_status < 2:
            # Only add bounding box if it's not largely occluded
            bounding_boxes.append((ped.id, "pedestrian",
                                (min_x, min_y, xdiff, ydiff), bbox[0]))
            #print(f"Pedestrian {ped.id} is visible, adding bounding box.")
        #else:
            #print(f"Pedestrian {ped.id} is largely occluded, skipping bounding box.")
    return bounding_boxes

def check_occlusion(points, z_values, depth_map, min_x, min_y, width, height):
    """
    Determines if an object is occluded based on depth information.
    
    Parameters:
    points (list): List of 2D points representing the bounding box corners
    z_values (np.ndarray): Z values for each corner point (distance from camera)
    depth_map (np.ndarray): Depth map from the camera
    min_x, min_y, width, height: Bounding box coordinates
    
    Returns:
    int: Occlusion level (0: fully visible, 1: partly occluded, 2: largely occluded, 3: unknown)
    """
    # Bound check to prevent out-of-bounds indexing
    h, w = depth_map.shape[:2]
    
    # Clip bounding box to image dimensions
    min_x_c = max(0, min(min_x, w-1))
    min_y_c = max(0, min(min_y, h-1))
    max_x_c = max(0, min(min_x + width, w-1))
    max_y_c = max(0, min(min_y + height, h-1))
    
    # If bounding box is outside the image, return unknown occlusion
    if min_x_c >= max_x_c or min_y_c >= max_y_c:
        return 3  # Unknown
        
    # Calculate expected depth of the object
    valid_z = [z for z in z_values if z > 0]
    if not valid_z:
        return 3  # Unknown - no valid depth values
        
    expected_depth = np.mean(valid_z)
    
    # Sample points within the bounding box
    visible_vertices = 0
    total_sampled_points = 0
    occluded_points = 0
    
    # Check occlusion for each vertex
    for i, point in enumerate(points):
        x, y = point
        if 0 <= x < w and 0 <= y < h:
            # Get actual depth at this pixel
            actual_depth = depth_map[y, x]
            #print(f"Point {i}: ({x}, {y}), Actual Depth: {actual_depth}, Expected Depth: {expected_depth}")
            
            # If the actual depth is significantly less than the expected depth, the vertex is occluded
            if actual_depth > 0 and actual_depth < z_values[i] - 1.0:
                occluded_points += 1
            else:
                visible_vertices += 1
                
            total_sampled_points += 1
    
    # Also sample some points inside the bounding box
    num_samples = min(25, (max_x_c - min_x_c) * (max_y_c - min_y_c) // 20)
    if num_samples > 0:
        x_samples = np.random.randint(min_x_c, max_x_c, num_samples)
        y_samples = np.random.randint(min_y_c, max_y_c, num_samples)
        
        for i in range(num_samples):
            x, y = x_samples[i], y_samples[i]
            actual_depth = depth_map[y, x]
            
            # We use expected_depth for interior points
            if actual_depth > 0 and actual_depth < expected_depth - 1.0:
                occluded_points += 1
                
            total_sampled_points += 1
    
    # Calculate occlusion percentage
    if total_sampled_points > 0:
        occlusion_percentage = occluded_points / total_sampled_points
        
        # Determine occlusion level
        if occlusion_percentage < 0.5:
            return 0  # Fully visible
        elif occlusion_percentage < 0.8:
            return 1  # Partly occluded
        else:
            return 2  # Largely occluded
    
    # If we can't determine occlusion level due to insufficient data
    if visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER:
        return 0  # Assume visible if we have enough visible vertices
    else:
        return 3  # Unknown

def get_2d_bounding_box(points):
    sorted_points = sort_points_clockwise(points)

    min_x = min(point[0] for point in sorted_points)
    min_y = min(point[1] for point in sorted_points)
    max_x = max(point[0] for point in sorted_points)
    max_y = max(point[1] for point in sorted_points)

    return int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)

def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def get_bboxes_3d(sensor: Any, point_cloud: np.ndarray, world: Any) -> List:
    vehicles = world.get_actors().filter("vehicle.*")
    pedestrians = world.get_actors().filter("walker.*")

    bounding_boxes = []

    for vehicle in vehicles:
        datapoint = create_kitti_datapoint_lidar(vehicle, sensor, point_cloud)
        if datapoint:
            bounding_boxes.append(datapoint)

    for pedestrian in pedestrians:
        datapoint = create_kitti_datapoint_lidar(pedestrian, sensor, point_cloud)
        if datapoint:
            bounding_boxes.append(datapoint)

    return bounding_boxes

def create_kitti_datapoint_lidar(actor, sensor, point_cloud):
    """
    Creates a KITTI format datapoint for an actor (vehicle or pedestrian) detected by the LiDAR sensor.
    
    Parameters:
    actor (carla.Actor): The vehicle or pedestrian actor
    sensor (carla.Sensor): The LiDAR sensor
    point_cloud (np.ndarray): The LiDAR point cloud data
    
    Returns:
    KittiDescriptor: KITTI format descriptor for the actor, or None if the actor isn't visible
    """
    # Get actor's bounding box in LiDAR sensor space
    bbox = ClientSideBoundingBoxes.get_bounding_box_lidar(actor, sensor)
    
    # Check if the bounding box is within sensor's view (all points have positive x value in LiDAR space)
    if not np.all(bbox[0, :] > 0):  # Use np.all() instead of all()
        return None
    
    # Filter points that are inside the bounding box
    points_in_bbox = points_in_bounding_box(point_cloud, bbox)
    
    # If no points detected, the object is not visible to LiDAR
    if len(points_in_bbox) < 1:  # Threshold can be adjusted
        return None
    
    # Create KITTI descriptor
    kitti_datapoint = KittiDescriptor()
    
    # Set object type - convert to lowercase for KITTI format
    obj_type = actor.attributes.get('role_name', 'car')
    if 'walker' in actor.type_id:
        obj_type = 'pedestrian'
    elif 'vehicle' in actor.type_id:
        if 'bicycle' in actor.type_id:
            obj_type = 'bicycle'
        elif 'motorcycle' in actor.type_id:
            obj_type = 'motorcycle'
        elif 'bus' in actor.type_id:
            obj_type = 'bus'
        elif 'truck' in actor.type_id:
            obj_type = 'truck'
        elif 'van' in actor.type_id:
            obj_type = 'van'
        else:
            obj_type = 'car'
    
    kitti_datapoint.set_type(obj_type.lower())
    
    # Set object ID
    kitti_datapoint.set_object_id(actor.id)
    
    # Set 3D dimensions
    kitti_datapoint.set_3d_object_dimensions(actor.bounding_box.extent)
    
    # Get actor's location in LiDAR sensor space
    location = actor.get_transform().location
    sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    
    # Convert actor location from world to sensor space
    location_world = np.array([[location.x], [location.y], [location.z], [1]])
    location_sensor = np.dot(world_sensor_matrix, location_world)
    
    actor_location = carla.Location(
        x=location_sensor[0, 0],
        y=location_sensor[1, 0],
        z=location_sensor[2, 0]
    )
    
    # Set location in KITTI format
    kitti_datapoint.set_3d_object_location(actor_location)
    
    # Set rotation (yaw angle)
    actor_rotation = actor.get_transform().rotation
    sensor_rotation = sensor.get_transform().rotation
    
    # Calculate relative rotation
    rotation_y = (actor_rotation.yaw - sensor_rotation.yaw) * np.pi / 180.0
    
    # Normalize to [-pi, pi]
    while rotation_y > np.pi:
        rotation_y -= 2 * np.pi
    while rotation_y < -np.pi:
        rotation_y += 2 * np.pi
    
    kitti_datapoint.set_rotation_y(rotation_y)
    
    return kitti_datapoint

def points_in_bounding_box(points, bbox):
    """
    Filters points that are inside a 3D bounding box.
    
    Parameters:
    points (np.ndarray): LiDAR point cloud data
    bbox (np.ndarray): 3D bounding box corners in sensor space
    
    Returns:
    np.ndarray: Points that are inside the bounding box
    """
    # Calculate min and max coordinates of the bounding box
    min_x = np.min(bbox[0, :])
    max_x = np.max(bbox[0, :])
    min_y = np.min(bbox[1, :])
    max_y = np.max(bbox[1, :])
    min_z = np.min(bbox[2, :])
    max_z = np.max(bbox[2, :])
    
    # Filter points inside the bounding box
    indices = np.where(
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )[0]
    
    return points[indices]


