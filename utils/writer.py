import os
import carla
import numpy as np
import open3d as o3d
import json
import cv2

from utils.bounding_boxes import get_bboxes_2d, get_bboxes_3d
from typing import Any, Dict, List

class Writer:
    def __init__(self, out_path: str, active_sensors: List) -> None:
        self.out_path = out_path
        self.sensors = active_sensors
        self.lidar_buffer = {}
        self._init_directory()

    def _init_directory(self) -> None:
        print("# ===== Creating Directories ========= #")
        for sensor in self.sensors:
            sensor_name = sensor.get_name()
            sensor_path = os.path.join(self.out_path, sensor_name)
            self.create_dir(sensor_path)
        print("# ===== Directories Created ========== #")

    def write_frame(self, data: Dict, frame: int, world: Any):

        for sensor in self.sensors:
            sensor_name = sensor.get_name()
            sensor_type = sensor.get_type()
            sensor_path = os.path.join(self.out_path, sensor_name)

            try:
                sensor_data = data[sensor_name]
            except:
                print(f"Writer Error: {sensor_name} data not found.")

            if sensor_data is not None:
                if sensor_type == "sensor.camera.rgb":
                    self.write_rgb(sensor.get_obj(), sensor_path, sensor_data, frame, world)
                elif sensor_type == "sensor.camera.dvs":
                    self.write_events(sensor.get_obj(), sensor_path, sensor_data, frame, world)
                elif sensor_type == "sensor.lidar.ray_cast":
                    self.write_lidar(sensor.get_obj(), sensor_path, sensor_data, frame, world)
                else:
                    raise ValueError(f"Unsupported sensor type: {sensor_type}")
                
    def generate_events_image(self, events: np.ndarray, width: int, height: int) -> np.ndarray:
        events_image = np.ones(
            (height, width, 3), dtype=np.uint8
        )
        events_image *= 255
        w_coords = events[:, 0].astype(np.int32)
        h_coords = events[:, 1].astype(np.int32)

        # Generate colors
        colors = np.zeros((events.shape[0], 3), dtype=np.uint8)
        polarities = events[:, 3]
        red = np.array([255, 0, 0])
        blue = np.array([0, 0, 255])
        colors[polarities >= 1] = red
        colors[polarities < 1] = blue

        # Generate image
        events_image[h_coords, w_coords, :] = colors
        events_image = events_image.astype(np.uint8)
        return events_image

    def write_rgb(self, sensor: Any, sensor_path: str, sensor_data: np.ndarray, frame: int, world: Any) -> None:
        (image, depth) = sensor_data

        frame_str = f"{frame:08d}"
        img_output_file = os.path.join(sensor_path, f"{frame_str}_img.png")
        bbox_output_file = os.path.join(sensor_path, f"{frame_str}_bboxes.json")

        height = image.shape[0]
        width = image.shape[1]

        bboxes = get_bboxes_2d(sensor, depth, world)

        cv2.imwrite(img_output_file, image)
        self.save_coco_format(bboxes, bbox_output_file, frame, f"{frame_str}_img.png", width, height)

    def write_events(self, sensor: Any, sensor_path: str, sensor_data: np.ndarray, frame: int, world: Any) -> None:
        (events, depth) = sensor_data

        frame_str = f"{frame:08d}"
        events_output_file = os.path.join(sensor_path, f"{frame_str}_events.npz")
        img_output_file = os.path.join(sensor_path, f"{frame_str}_events_img.png")
        bbox_output_file = os.path.join(sensor_path, f"{frame_str}_bboxes.json")

        width = int(sensor.attributes.get("image_size_x"))
        height = int(sensor.attributes.get("image_size_y"))

        events_image = self.generate_events_image(events, width, height)
        bboxes = get_bboxes_2d(sensor, depth, world)

        np.savez_compressed(events_output_file, dvs_events=events)
        cv2.imwrite(img_output_file, events_image)
        self.save_coco_format(bboxes, bbox_output_file, frame, f"{frame_str}_events_img.png", width, height)


    def write_lidar(self, sensor: Any, sensor_path: str, point_cloud: np.ndarray, frame: int, world: Any) -> None:
        frame_str = f"{frame:08d}"
        pcd_output_file = os.path.join(sensor_path, f"{frame_str}_cloud.pcd")
        bbox_output_file = os.path.join(sensor_path, f"{frame_str}_bboxes.txt")

        points = point_cloud[:, :-1]
        intensities = point_cloud[:, -1]
        
        bboxes = get_bboxes_3d(sensor, points, world)

        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
        pcd.point["intensity"] = o3d.core.Tensor(intensities.reshape(-1, 1), o3d.core.float32)
        o3d.t.io.write_point_cloud(pcd_output_file, pcd, write_ascii=True)

        self.save_kitti_3d_format(bboxes, bbox_output_file)

    def save_coco_format(self, bounding_boxes, file_path, id, image_filename, image_width, image_height) -> None:
        coco = {
            "car": 1,
            "truck": 2,
            "van": 3,
            "pedestrian": 4,
            "motorcycle": 5,
            "bicycle": 6,
            "Bus": 7
        }
        
        coco_data = {
            "images": [
                {
                    "id": id,
                    "file_name": image_filename,
                    "width": image_width,
                    "height": image_height,
                }
            ],
            "annotations": [
            ],
            "categories": [
                {"id": 1, "name": "car", "supercategory": "vehicle"},
                {"id": 2, "name": "truck", "supercategory": "vehicle"},
                {"id": 3, "name": "van", "supercategory": "vehicle"},
                {"id": 4, "name": "pedestrian", "supercategory": "human"},
                {"id": 5, "name": "motorcycle", "supercategory": "vehicle"},
                {"id": 6, "name": "bicycle", "supercategory": "vehicle"},
                {"id": 7, "name": "bus", "supercategory": "vehicle"}
            ]
        }
        for obj_id, class_name, bbox_2d, bbox_3d in bounding_boxes:
            coco_data["annotations"].append({
                "id": obj_id,
                "image_id": id,
                "category_id": coco[class_name],
                "bbox_2d": bbox_2d,
                "area": bbox_2d[2] * bbox_2d[3],
                "bbox_3d": bbox_3d,
                "iscrowd": 0,
                "segmentation": [],
            })
        with open(file_path, 'w') as file:
            json.dump(coco_data, file, indent=4)

    def save_kitti_3d_format(self, annotations: List, filepath: str) -> None:
        with open(filepath, "w") as file:
            for element in annotations:
                file.write(str(element) + "\n")

    def create_dir(self, path: str) -> str:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory created: ", path)
        return path