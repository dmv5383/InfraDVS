import numpy as np
import cv2

from typing import Any, Tuple

def extract_events(data: Any, sim_time: float) -> Tuple[np.ndarray, np.ndarray]:
    (events_data, depth_data) = data
    raw_data = np.frombuffer(events_data.raw_data, dtype=([
        ("x", np.uint16),
        ("y", np.uint16),
        ("t", np.int64),
        ("pol", bool)
    ]))

    time = int(sim_time*1e6)

    events = np.zeros((raw_data[:]["x"].shape[0], 4), dtype=np.float64)
    events[:, 0] = raw_data[:]["x"]
    events[:, 1] = raw_data[:]["y"]
    events[:, 2] = sim_time*1e6
    events[:, 3] = raw_data[:]["pol"]
    events = events.astype(np.int64)

    depth = extract_depth(depth_data)

    return (events, depth)

def extract_rgb(data: Any, sim_time: float) -> Tuple[np.ndarray, np.ndarray]:
    (image_data, depth_data) = data
    raw_data = np.frombuffer(image_data.raw_data, dtype=np.uint8)

# TODO: Check if data or raw_data
    rgb = np.reshape(raw_data, (image_data.height, image_data.width, 4))
    rgb = rgb[:, :, :3]
#    rgb = rgb[:, :, ::-1]

    time = int(sim_time*1e6)

    depth = extract_depth(depth_data)

    return (rgb, depth)

def extract_lidar(data: Any, sim_time: float) -> np.ndarray:
    raw_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))

    point_cloud = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    time = int(sim_time*1e6)

    return point_cloud

def extract_depth(data: Any) -> np.ndarray:
    raw_data = np.frombuffer(data.raw_data, dtype=np.uint8)
    depth = np.reshape(raw_data, (data.height, data.width, 4))
    depth = depth[:, :, :3]
    depth = depth[:, :, ::-1]

    # Normalize depth values
    R = depth[:, :, 0].astype(np.float32)
    G = depth[:, :, 1].astype(np.float32)
    B = depth[:, :, 2].astype(np.float32)
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    normalized_meters = normalized * 1000
    return normalized_meters