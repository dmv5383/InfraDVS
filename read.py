import carla

from reader import ScenarioReader

from typing import Any, Dict, Tuple

class ReadScenario():
    def __init__(
        self,
        client: Any,
        out_path: str,
        sensors: Dict[str, Any] = None,
        start_time: float = 0.5,
        tick_rate: float = 0.1,
        world_map: str = None,
        world_weather: str = None,
        record_path: str = None,
        record_delta_time: float = 60.0
    ) -> None:
        scenario_reader = ScenarioReader(
            client=client,
            out_path=out_path,
            sensors=sensors,
            start_time=start_time,
            tick_rate=tick_rate,
            record_path=record_path,
            record_delta_time=record_delta_time,
            world_map=world_map,
            world_weather=world_weather,
            client_timeout=20.0,
            init_sleep=0.0
        )
        scenario_reader.loop()