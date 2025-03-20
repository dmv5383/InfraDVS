import os
import h5py
import hdf5plugin
import numpy as np
import open3d as o3d

from typing import Any, Dict, List, Tuple, Callable, Union

class BaseWriter:
    def __init__(self, sensor_name: str, out_path: str) -> None:
        self.sensor_name = sensor_name
        self.out_path = out_path
        self.compressor = hdf5plugin.Blosc(cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE)

class RGBWriter(BaseWriter):
    def __init__(self, sensor_name: str, out_path: str) -> None:
        super().__init__(sensor_name, out_path)
        self._init_file()

    def _init_file(self) -> None:
        self.rgb_path = os.path.join(self.out_path, self.sensor_name + "_rgb.hdf5")
        self.rgb_file = h5py.File(self.rgb_path, "a")
        self.rgb_flag = False

    def write(self, data: np.ndarray, time: float, vehicle_bounding_boxes: np.ndarray, pedestrian_bounding_boxes: np.ndarray) -> None:
        rgb_image = data[None, ...].astype(np.uint8)

        if self.rgb_flag is False:
            self.time_offset = 0
            if time != 0:
                self.time_offset = time
            self._save_time_offset(data_file=self.rgb_file, time=self.time_offset)

            self.rgb_images = self.rgb_file.create_dataset(
                name="rgb_images", data=rgb_image,
                chunks=True, maxshape=(None, *rgb_image.shape[1:]), dtype=np.uint8,
                **self.compressor
            )
            time: np.ndarray = np.array(time, dtype=np.int64)[None, ...]
            self.rgb_time = self.rgb_file.create_dataset(
                name="time", data=time - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.rgb_vehicle_bboxes = self.rgb_file.create_dataset(
                name="vehicle_bounding_boxes", data=vehicle_bounding_boxes[None, ...],
                chunks=True, maxshape=(None, *vehicle_bounding_boxes.shape), dtype=np.float32,
                **self.compressor
            )
            self.rgb_ped_bboxes = self.rgb_file.create_dataset(
                name="pedestrian_bounding_boxes", data=pedestrian_bounding_boxes[None, ...],
                chunks=True, maxshape=(None, *pedestrian_bounding_boxes.shape), dtype=np.float32,
                **self.compressor
            )
            self.rgb_flag = True

        else:
            data_points = rgb_image.shape[0]
            dataset_points = self.rgb_images.shape[0]
            all_points = data_points + dataset_points
            self.rgb_images.resize(all_points, axis=0)
            self.rgb_images[-data_points:] = rgb_image
            self.rgb_time.resize(all_points, axis=0)
            self.rgb_time[-data_points:] = time - self.time_offset
            self.rgb_vehicle_bboxes.resize(all_points, axis=0)
            self.rgb_vehicle_bboxes[-data_points:] = vehicle_bounding_boxes[None, ...]
            self.rgb_ped_bboxes.resize(all_points, axis=0)
            self.rgb_ped_bboxes[-data_points:] = pedestrian_bounding_boxes[None, ...]

    def _save_time_offset(self, data_file: h5py.File, time: int) -> None:
        time = np.array(time, dtype=np.int64)[None, ...]
        data_file.create_dataset(
            name="time_offset", data=time,
            chunks=True, maxshape=(1,), dtype=np.int64,
            **self.compressor
        )

class DVSWriter(BaseWriter):
    def __init__(self, sensor_name: str, out_path: str) -> None:
        super().__init__(sensor_name, out_path)
        self._init_file()

    def _init_file(self) -> None:
        self.events_path = os.path.join(self.out_path, self.sensor_name + "_events.hdf5")
        self.events_file = h5py.File(self.events_path, "a")
        self.events_flag = False

        self.events_group = self.events_file.create_group("events")

    def write(self, data: np.ndarray, time: float, vehicle_bounding_boxes: np.ndarray, pedestrian_bounding_boxes: np.ndarray) -> None:
        if self.events_flag is False:
            self.time_offset = 0
            if data[0, 2] != 0:
                self.time_offset = data[0, 2]
            self._save_time_offset(
                data_file=self.events_file, time=self.time_offset
            )

            self.events_x = self.events_group.create_dataset(
                name="x", data=data[:, 0],
                chunks=True, maxshape=(None,), dtype=np.uint16,
                **self.compressor
            )
            self.events_y = self.events_group.create_dataset(
                name="y", data=data[:, 1],
                chunks=True, maxshape=(None,), dtype=np.uint16,
                **self.compressor
            )
            self.events_time = self.events_group.create_dataset(
                name="time", data=data[:, 2] - self.time_offset,
                chunks=True, maxshape=(None,), dtype=np.int64,
                **self.compressor
            )
            self.events_pol = self.events_group.create_dataset(
                name="polarity", data=data[:, 3],
                chunks=True, maxshape=(None,), dtype=bool,
                **self.compressor
            )
            self.events_vehicle_bboxes = self.events_file.create_dataset(
                name="vehicle_bounding_boxes", data=vehicle_bounding_boxes[None, ...],
                chunks=True, maxshape=(None, *vehicle_bounding_boxes.shape), dtype=np.float32,
                **self.compressor
            )
            self.events_ped_bboxes = self.events_file.create_dataset(
                name="pedestrian_bounding_boxes", data=pedestrian_bounding_boxes[None, ...],
                chunks=True, maxshape=(None, *pedestrian_bounding_boxes.shape), dtype=np.float32,
                **self.compressor
            )
            self.events_flag = True
        else:
            data_points = data.shape[0]
            dataset_points = self.events_x.shape[0]
            all_points = data_points + dataset_points
            self.events_x.resize(all_points, axis=0)
            self.events_x[-data_points:] = data[:, 0]
            self.events_y.resize(all_points, axis=0)
            self.events_y[-data_points:] = data[:, 1]
            self.events_time.resize(all_points, axis=0)
            self.events_time[-data_points:] = data[:, 2] - self.time_offset
            self.events_pol.resize(all_points, axis=0)
            self.events_pol[-data_points:] = data[:, 3]
            self.events_vehicle_bboxes.resize(all_points, axis=0)
            self.events_vehicle_bboxes[-data_points:] = vehicle_bounding_boxes[None, ...]
            self.events_ped_bboxes.resize(all_points, axis=0)
            self.events_ped_bboxes[-data_points:] = pedestrian_bounding_boxes[None, ...]

    def _save_time_offset(self, data_file: h5py.File, time: int) -> None:
        time = np.array(time, dtype=np.int64)[None, ...]
        data_file.create_dataset(
            name="time_offset", data=time,
            chunks=True, maxshape=(1,), dtype=np.int64,
            **self.compressor
        )

class LidarWriter(BaseWriter):
    def __init__(self, sensor_name: str, out_path: str) -> None:
        super().__init__(sensor_name, out_path)
        self._init_file()

    def _init_file(self) -> None:
        self.lidar_path = os.path.join(self.out_path, self.sensor_name + "_lidar")
        if not os.path.exists(self.lidar_path):
            os.makedirs(self.lidar_path)

    def write(self, data: np.ndarray, time: float, vehicle_bounding_boxes: np.ndarray, pedestrian_bounding_boxes: np.ndarray) -> None:
        points = data[:,:-1]
        intensities = data[:,-1]

        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
        pcd.point["intensity"] = o3d.core.Tensor(intensities.reshape(-1, 1), o3d.core.float32)
        filename = self.lidar_path + "/lidar_{}.pcd".format(time)
        o3d.t.io.write_point_cloud(filename, pcd, write_ascii=True)

        vehicle_bbox_filename = self.lidar_path + "/vehicle_bounding_boxes_{}.txt".format(time)
        np.savetxt(vehicle_bbox_filename, vehicle_bounding_boxes, fmt='%f')
        ped_bbox_filename = self.lidar_path + "/vehicle_bounding_boxes_{}.txt".format(time)
        np.savetxt(ped_bbox_filename, pedestrian_bounding_boxes, fmt='%f')