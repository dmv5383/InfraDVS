import carla
import time

from base import ScenarioBase
from utils.sensor import Sensor
from utils.sync import SensorSync
from utils.spawn import VehicleSpawner, TrafficSpawner
from utils import extract

from datetime import timedelta
from typing import Any, Dict, List, Tuple

from utils.writer import Writer
from tqdm import tqdm

class ScenarioCreator(ScenarioBase):
    def __init__(
            self,
            client: Any,
            out_path: str,
            record_path:str,
            vehicles: List[Dict],
            num_traffic_vehicles: int = None,
            num_traffic_peds: int = None,
            start_time: float = 1.0,
            tick_rate: float = 0.1,
            record_start_time: float = 5.0,
            record_delta_time: float = 60.0,
            **kwargs
    ) -> None:
        super().__init__(
            client=client, simulation_mode="create", out_path=out_path, start_time=start_time,
            tick_rate=tick_rate, **kwargs
        )
        self.record_path = record_path
        self.vehicles = vehicles
        self.num_traffic_vehicles = num_traffic_vehicles
        self.num_traffic_peds = num_traffic_peds
        self.record_start_time = record_start_time
        self.record_delta_time = record_delta_time
        self._init_vehicles()
        self._init_sensors()
        self._init_writer()
        self._save_sensor_props()
        
    def _init_vehicles(self) -> None:
        self.vehicle_spawner = VehicleSpawner(client=self.client, world=self.world)
        self.vehicle_spawner.spawn_vehicles(self.vehicles)

        self.traffic_spawner = TrafficSpawner(
            client=self.client,
            world=self.world,
            num_vehicles=self.num_traffic_vehicles,
            num_peds=self.num_traffic_peds
        )

        for vehicle in self.vehicle_spawner.all_vehicles:
            vehicle.set_autopilot(True, self.traffic_spawner.traffic_manager.get_port())
    
    def loop(self) -> None:
        try:
            self.sim_time = 0.0
            self.real_time = time.time()

            # Initialize progress bar
            initial_frame = self.world.get_snapshot().frame
            end_frame = initial_frame + int(self.record_delta_time / self.tick_rate)
            progress_bar = tqdm(total=end_frame - initial_frame, desc="Simulation Progress", unit="frame")

            with SensorSync(
                world=self.world, sensors=self.active_sensors,
                tick_rate=self.tick_rate, start_time=self.start_time
            ) as sensor_sync:
                self.record_flag = False
                self.end_record_flag = False

                while True:
                    # Start recording
                    if self.sim_time > self.record_start_time and not self.record_flag and not self.end_record_flag:
                        self.client.start_recorder(self.record_path, True)
                        self.record_flag = True

                    # Stop recording
                    if self.sim_time > (self.record_start_time + self.record_delta_time) and self.record_flag and not self.end_record_flag:
                        self.client.stop_recorder()
                        self.end_record_flag = True
                        self.record_flag = False
                        break

                    # Parse data
                    data = sensor_sync.tick(timeout=2.0)
                    data = self._extract_data(data)

                    # Update status
                    self._print_status(
                        frame=data["world"],
                        end_frame=end_frame,
                        data=data,
                        sim_time=self.sim_time,
                        real_time=time.time() - self.real_time,
                        progress_bar=progress_bar
                    )

                    # Save data
                    self._save_data(data, data["world"], self.world)

                    # Update simulation time
                    self.sim_time += self.tick_rate
        finally:
            progress_bar.close()
            self._reset_settings()
            self.vehicle_spawner.destroy_vehicles()
            self._destroy_sensors()
            self.traffic_spawner.destroy_traffic()
            print("All simulation elements reset.")
