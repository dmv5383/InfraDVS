import carla
import os
import argparse
import json

from read import ReadScenario

def parse_args():
    parser = argparse.ArgumentParser(description="Run CARLA scenario with specified parameters.")
    parser.add_argument("--world_map", type=str, required=True, help="World map to load in CARLA.")
    parser.add_argument("--world_weather", type=str, required=True, help="Weather preset to use in CARLA.")
    parser.add_argument("--record_delta_time", type=int, required=True, help="Delta time for recording.")
    parser.add_argument("--out_path", type=str, default="/home/dmv5383/CARLA_DVS_Scripts/InfraDVS/datasets/data/", help="Output path for data.")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time for the scenario.")
    parser.add_argument("--tick_rate", type=float, default=0.1, help="Tick rate for the scenario.")
    parser.add_argument("--sensors_config", type=str, required=True, help="Path to the JSON file containing sensor configuration.")
    parser.add_argument("--record_path", type=str, required=True, help="Path to the recorded scenario log.")
    return parser.parse_args()

def load_sensors_config(file_path):
    with open(file_path, 'r') as file:
        sensors_data = json.load(file)
    sensors = []
    for sensor in sensors_data:
        sensor["transform"] = carla.Transform(
            carla.Location(**sensor["transform"]["location"]),
            carla.Rotation(**sensor["transform"]["rotation"])
        )
        sensors.append(sensor)
    return sensors

if __name__ == "__main__":
    args = parse_args()

    client = carla.Client("localhost", 2000)
    world_map = args.world_map
    world_weather = args.world_weather
    record_path = args.record_path
    record_delta_time = args.record_delta_time
    out_path = args.out_path
    base_name = os.path.basename(record_path)
    file_name = os.path.splitext(base_name)[0]
    file_path = os.path.join(out_path, file_name)
    start_time = args.start_time
    tick_rate = args.tick_rate

    sensors = load_sensors_config(args.sensors_config)

    read_scenario = ReadScenario(
        client=client, out_path=file_path, sensors=sensors, start_time=start_time,
        tick_rate=tick_rate, world_map=world_map, world_weather=world_weather,
        record_path=record_path, record_delta_time=record_delta_time
    )