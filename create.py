import carla

from creator import ScenarioCreator

from typing import Any, Tuple, List, Dict

class CreateScenario():
    def __init__(
        self,
        client: Any,
        out_path: str,
        record_path: str,
        world_map: str,
        world_weather: str,
        vehicles: List[Dict],
        num_traffic_vehicles: int = None,
        num_traffic_peds: int = None,
        record_start_time: float = 5.0,
        record_delta_time: float = 60.0
    ) -> None:
        
        self.sensors = [
            {
                "name": "car_lidar",
                "type": "sensor.lidar.ray_cast",
                "options": {
                    "upper_fov": "21.2",
                    "lower_fov": "-21.2",
                    "channels": "128",
                    "range": "200",
                    "points_per_second": "5242880",
                    "rotation_frequency": "10"
                },
                "transform": carla.Transform(
                    carla.Location(x=2.8, z=1.8), 
                    carla.Rotation(pitch=15)
                ),
                "converter": None,
                "actor": "ego_vehicle",
                "is_static": False
            }
        ]
        
        # Run scenario
        scenario_creator = ScenarioCreator(
            client=client,
            out_path=out_path,
            record_path=record_path,
            start_time=record_start_time,
            tick_rate=0.001,
            world_map=world_map,
            world_weather=world_weather,
            client_timeout=5.0,
            init_sleep=0.0,
            vehicles=vehicles,
            sensors=self.sensors,
            record_start_time=record_start_time,
            record_delta_time=record_delta_time,
            num_traffic_vehicles=num_traffic_vehicles,
            num_traffic_peds=num_traffic_peds
        )
        scenario_creator.loop()