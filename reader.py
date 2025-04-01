import os
import time
import carla
import json
import traceback

from base import ScenarioBase
from utils.sync import SensorSync

from utils import extract
from utils.writer import Writer

from datetime import timedelta
from typing import Any, Dict, Tuple
from tqdm import tqdm

class ScenarioReader(ScenarioBase):
    def __init__(
            self,
            client: Any,
            out_path: str,
            start_time: float = 0.5,
            tick_rate: float = 0.1,
            record_path: str = None,
            record_delta_time: float = 60.0,
            **kwargs
    ) -> None:
        super().__init__(
            client=client, simulation_mode="read", out_path=out_path,
            start_time=start_time, tick_rate=tick_rate, **kwargs
        )
        self.record_path = record_path
        self.record_delta_time = record_delta_time
        self._read_recording()
        self._init_sensors()
        self._init_writer()
        self._save_sensor_props()
        time.sleep(10)

    def _read_recording(self) -> None:
        self.client.replay_file(self.record_path, 0, 0, 0, False) # TODO: Check parameters
        self.world.tick()

    def _destroy_actors(self) -> None:
        for actor in self.world.get_actors():
            actor.destroy()
    
    def _map_data(self, ) -> None:
        # TODO: Create mapping for different writers
        pass

    def create_dir(self, path: str) -> str:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory created: ", path)
        return path

    def loop(self) -> None:
        try:
            self.sim_time = 0.0
            self.real_time = time.time()

            # Initialize progress bar
            initial_frame = self.world.get_snapshot().frame
            end_frame = initial_frame + int(self.record_delta_time / self.tick_rate)
            progress_bar = tqdm(total=end_frame - initial_frame, desc="Replay Progress", unit="frame")

            with SensorSync(
                world=self.world, sensors=self.active_sensors,
                start_time=self.start_time, tick_rate=self.tick_rate
            ) as sensor_sync:
                while True:
                    data = sensor_sync.tick(timeout=5.0)
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
                    if self.sim_time > (self.record_delta_time - self.start_time):
                        break
        except Exception as e:
            print("Error:", e)
            print(traceback.format_exc())
        finally:
            progress_bar.close()
            self._reset_settings()
            self._destroy_sensors()
            self._destroy_actors()
            print("Actors destroyed.")
            print("All simulation elements reset.")