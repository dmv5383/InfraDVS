import queue
from queue import Empty
from typing import Any, Dict

import numpy as np
import carla

class Sensor():
    def __init__(
            self,
            world: Any,
            sensor: Dict[str, Any],
            tick_rate: float = 0.1
    ) -> None:
        self.world = world
        self.sensor = sensor
        self.tick_rate = tick_rate

        self.world_bp = self.world.get_blueprint_library()
        self._init_sensor()

    def _init_sensor(self) -> None:
        # Sensor configuration
        self.name: str = self.sensor["name"]
        self.type: str = self.sensor["type"]
        self.options: Dict[str, Any] = self.sensor["options"]
        self.transform: Any = self.sensor["transform"]
        self.is_static: bool = self.sensor["is_static"]
        if not self.is_static:
            for actor in self.world.get_actors().filter("vehicle.*"):
                if actor.attributes["role_name"] == self.sensor["actor"]:
                    self.actor: Any = actor
                    break
        self.converter: Any = self.sensor["converter"]

        # Sensor synchronization
        self.sync_flag = False
        self.frame_count = 1
        if "sensor_tick" in self.options.keys():
            self.parsing_freq = int(float(self.options["sensor_tick"])/self.tick_rate)
            if self.parsing_freq == 0:
                self.parsing_freq = 1
        else:
            self.parsing_freq = 1

        # Spawn sensor
        self.sensor_bp = self.world_bp.find(self.type)
        for option in self.options.keys():
            self.sensor_bp.set_attribute(option, self.options[option])

        if self.type == "sensor.lidar.ray_cast" and not self.is_static:
            hp = max(self.actor.bounding_box.extent.x,self.actor.bounding_box.extent.y)*np.tan(np.radians(-float(self.sensor_bp.get_attribute("lower_fov"))))
            self.transform = carla.Transform(carla.Location(z=2*self.actor.bounding_box.extent.z+hp))

        if self.is_static:
            self.sensor_obj = self.world.spawn_actor(
                self.sensor_bp, self.transform
            )
        else:
            self.sensor_obj = self.world.spawn_actor(
                self.sensor_bp, self.transform, attach_to=self.actor
            )

    def _parse_data(
            self,
            world_frame: Any,
            sensor_queue: queue.Queue,
            timeout: float = 2.0
    ) -> Any:
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == world_frame:
                    return data
            except Empty:
                return None

    # === User Functions === #
    def get_name(self) -> str:
        """Returns sensor name.
        """
        return self.name
    
    def get_type(self) -> str:
        return self.type

    def get_bp(self) -> Any:
        return self.sensor_bp

    def get_obj(self) -> Any:
        """Returns sensor object.
        """
        return self.sensor_obj
    
    def get_is_static(self) -> bool:
       return self.is_static

    def read_data(
        self,
        world_frame: Any,
        sensor_queue: queue.Queue,
        timeout: float = 2.0
    ) -> Any:
        """User sensor data reading function.
        """
        data = None
        if self.sync_flag == False:
            data = self._parse_data(world_frame, sensor_queue, timeout)
            if data is not None:
                self.sync_flag = True
                self.frame_count = 1
        else:
            if self.frame_count % self.parsing_freq == 0:
                data = self._parse_data(world_frame, sensor_queue, timeout)
            self.frame_count += 1
        return data