import carla
import os
import argparse
from create import CreateScenario

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a CARLA scenario.")
    parser.add_argument("--world_map", type=str, default="Town10HD", help="The world map to use.")
    parser.add_argument("--world_weather", type=str, default="ClearNoon", help="The weather condition to use.")
    parser.add_argument("--run_number", type=int, default=1, help="The run number for the scenario.")
    parser.add_argument("--out_path", type=str, default="datasets/data/", help="The output path for the LIDAR sensor.")
    parser.add_argument("--record_start_time", type=int, default=10, help="How long into the simulation the recording begins.")
    parser.add_argument("--record_delta_time", type=int, default=60, help="The length of the recording.")
    parser.add_argument("--num_vehicles", type=int, default=20, help="The number of vehicles in the scenario.")
    parser.add_argument("--num_peds", type=int, default=20, help="The number of pedestrians in the scenario.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    client = carla.Client("localhost", 2000)
    world_map = args.world_map
    world_weather = args.world_weather
    run_number = args.run_number
    out_path = args.out_path
    file_name = f"{world_map}_{world_weather}_{run_number}"
    file_path = os.path.join(out_path, file_name)
    record_start_time = args.record_start_time
    record_delta_time = args.record_delta_time
    num_vehicles = args.num_vehicles
    num_peds = args.num_peds

    vehicles = [
        {"role_name": "ego_vehicle",
         "type": "vehicle.nissan.patrol"
         }
    ]

    # Create scenario

    #if not os.path.exists(out_path):
    #    os.makedirs(out_path)

    create_scenario = CreateScenario(
        client=client, out_path=file_path, world_map=world_map,
        world_weather=world_weather, vehicles=vehicles,
        num_traffic_vehicles=num_vehicles, num_traffic_peds=num_peds,
        record_start_time=record_start_time, record_delta_time=record_delta_time
    )