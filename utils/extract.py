import numpy as np

from typing import Any, Tuple

def extract_events(
    data: Any, sim_time: float
) -> np.ndarray:
    raw_data = np.frombuffer(data.raw_data, dtype=([
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

    return events

def extract_rgb(
    data: Any, sim_time: float
) -> np.ndarray:
    raw_data = np.frombuffer(data.raw_data, dtype=np.uint8)

# TODO: Check if data or raw_data
    rgb = np.reshape(raw_data, (data.height, data.width, 4))
    rgb = rgb[:, :, :3]
#    rgb = rgb[:, :, ::-1]

    time = int(sim_time*1e6)
    return rgb

def extract_lidar(
        data: Any, sim_time: float
) -> np.ndarray:
    raw_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))

    point_cloud = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    time = int(sim_time*1e6)

    return point_cloud