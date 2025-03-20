import carla
import time

from utils import extract
from utils.sensor import Sensor

from typing import Any, Dict, Tuple

from utils.writer import Writer

class ScenarioBase():
    def __init__(
        self,
        client: Any,
        simulation_mode: str,
        out_path: str,
        sensors: Dict[str, Any] = None,
        start_time: float = 0.5,
        tick_rate: float = 0.1,
        world_map: str = None,
        world_weather: str = None,
        client_timeout: float = 10.0,
        init_sleep: float = 10.0
    ) -> None:
        self.client = client
        self.simulation_mode = simulation_mode
        self.out_path = out_path
        self.sensors = sensors
        self.start_time = start_time
        self.tick_rate = tick_rate
        self.world_map = world_map
        self.world_weather = world_weather
        self.client_timeout = client_timeout
        self.init_sleep = init_sleep
        self._init_simulation()

    def _init_simulation(self) -> None:
        self.client.set_timeout(self.client_timeout)
        if self.world_map is not None:
            self.client.load_world(self.world_map)
        time.sleep(self.init_sleep)
        self.world = self.client.get_world()

        if self.simulation_mode == "create":
            settings = carla.WorldSettings(
                no_rendering_mode=True,
                synchronous_mode=True,
                fixed_delta_seconds=self.tick_rate)
            self.init_settings = self.world.get_settings()
            self.world.apply_settings(settings)
        else:
            settings = carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.tick_rate)
            self.init_settings = self.world.get_settings()
            self.world.apply_settings(settings)

        if self.world_weather is not None:
            self.world_weather = getattr(carla.WeatherParameters, self.world_weather)
            self.world.set_weather(self.world_weather)

    def _init_sensors(self) -> None:
        self.active_sensors = []
        print("# ===== Initializing Sensors ============= #")
        if self.sensors is not None:
            for sensor in self.sensors:
                self.active_sensors.append(Sensor(self.world, sensor, self.tick_rate))
                print("Initialized: ", sensor["name"], sensor["type"])
        print("# ===== Sensors Initialized ============== #")

    def _init_writer(self) -> None:
        self.writer = Writer(out_path=self.out_path, active_sensors=self.active_sensors)

    def _extract_data(self, data: Dict[str, Any]) -> None:
        for sensor in self.active_sensors:
            sensor_name = sensor.get_name()
            if data[sensor_name] is not None:
                sensor_type = sensor.get_type()

                if sensor_type == "sensor.camera.rgb":
                    data[sensor_name] = getattr(
                        extract, "extract_rgb"
                    )(data[sensor_name], sim_time=self.sim_time)
                elif sensor_type == "sensor.camera.dvs":
                    data[sensor_name] = getattr(
                        extract, "extract_events"
                    )(data[sensor_name], sim_time=self.sim_time)
                elif sensor_type == "sensor.lidar.ray_cast":
                    data[sensor_name] = getattr(
                        extract, "extract_lidar"
                    )(data[sensor_name], sim_time=self.sim_time)
                else:
                    raise ValueError(f"Unsupported sensor type: {sensor_type}")
        return data
    
    def _print_data(self, data: Dict[str, Any]) -> None:
        for sensor_name, info in data.items():
            print(sensor_name + ":", info is not None)

    def _save_data(self, data: Dict[str, Any], frame: int, world: Any) -> None:
        self.writer.write_frame(data=data, frame=frame, world=world)

    def loop(self) -> None:
        raise NotImplementedError

    def _reset_settings(self) -> None:
        self.world.apply_settings(self.init_settings)
        print('Reset Simulation Settings')

    def _destroy_sensors(self) -> None:
        for sensor in self.active_sensors:
            sensor.get_obj().destroy()
        print('Destroyed Sensors')