import queue

from utils.sensor import Sensor

from typing import Any, Dict, List


class SensorSync():
    def __init__(
        self,
        world: Any,
        sensors: List[Sensor],
        tick_rate: float,
        start_time: float
    ) -> None:
        self.world = world
        self.sensors = sensors
        self.tick_rate = tick_rate
        self.start_time = start_time
        self._init_sync()

    def _init_sync(self) -> None:
        self.sensors_queues = {}
        self.iter = 0
        self.start_iter = int(self.start_time/self.tick_rate)

    def tick(self, timeout: float = 2.0) -> Dict[str, Any]:
        # Initialize data
        data = {"world": None}
        for sensor in self.sensors:
            data.update({sensor.get_name(): None})

        # Tick simulation
        self.world_frame = self.world.tick()
        data["world"] = self.world_frame

        # Get sensors data
        if self.iter >= self.start_iter:
            for sensor in self.sensors:
                data[sensor.get_name()] = sensor.read_data(
                    world_frame=self.world_frame,
                    sensor_queue=self.sensors_queues[sensor.get_name()],
                    timeout=timeout
                )

        self.iter += 1
        return data
    
    def __enter__(self) -> None:

        def create_queue(name: str, on_tick: Any) -> None:
            s_queue = queue.Queue()
            on_tick(s_queue.put)
            self.sensors_queues.update({name: s_queue})

        create_queue(name="world", on_tick=self.world.on_tick)
        for sensor in self.sensors:
            create_queue(name=sensor.get_name(), on_tick=sensor.get_obj().listen)
        
        return self
    
    def __exit__(self, *args, **kwargs) -> None:
        pass

