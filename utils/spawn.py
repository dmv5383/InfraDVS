import carla
import numpy as np
import time
import random

from typing import Any, List

class VehicleSpawner():
    def __init__(self, client: Any, world: Any) -> None:
        self.client = client
        self.world = world
        self.map = self.world.get_map()
        self.settings = self.world.get_settings()

        self.all_vehicles = []

    def get_vehicles(self) -> List[Any]:
        return self.all_vehicles
    
    def destroy_vehicles(self) -> None:
        for vehicle in self.all_vehicles:
            vehicle.destroy()
        print("All vehicles destroyed.")

    def spawn_vehicles(
            self,
            vehicles: List[dict]
    ) -> None:
        self.world_bp_vehicles = self.world.get_blueprint_library().filter("vehicle.*")
        self.world_bp_vehicles = [
            vehicle for vehicle in self.world_bp_vehicles
            if int(vehicle.get_attribute("number_of_wheels")) == 4
        ]

        self.spawn_points = self.map.get_spawn_points()

        for i, vehicle_info in enumerate(vehicles):
            if i >= len(self.spawn_points):
                break

            if "spawn_point" in vehicle_info:
                point_index = vehicle_info["spawn_point"]
                point = self.spawn_points[point_index]
            else:
                point = self.spawn_points[i]

            vehicle_type = vehicle_info.get("type")
            if vehicle_type:
                vehicle_bp = self.world.get_blueprint_library().find(vehicle_type)
            else:
                vehicle_bp = np.random.choice(self.world_bp_vehicles)

            if vehicle_bp.has_attribute("color"):
                if "color" in vehicle_info:
                    vehicle_bp.set_attribute("color", vehicle_info["color"])
                else:
                    vehicle_color = random.choice(vehicle_bp.get_attribute("color").recommended_values)
                    vehicle_bp.set_attribute("color", vehicle_color)
                    
            if vehicle_bp.has_attribute("driver_id"):
                if "driver_id" in vehicle_info:
                    vehicle_bp.set_attribute("driver_id", vehicle_info["driver_id"])
                else:
                    vehicle_id = random.choice(vehicle_bp.get_attribute("driver_id").recommended_values)
                    vehicle_bp.set_attribute("driver_id", vehicle_id)
                    
            if "role_name" in vehicle_info:
                vehicle_bp.set_attribute("role_name", vehicle_info["role_name"])
            else:
                vehicle_bp.set_attribute("role_name", "autopilot")

            vehicle = self.world.try_spawn_actor(vehicle_bp, point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.all_vehicles.append(vehicle)

class TrafficSpawner():
    def __init__(
            self,
            client: Any,
            world: Any,
            num_vehicles: int = 20,
            num_peds: int = 30
    ) -> None:
        self.client = client
        self.world = world
        self.map = self.world.get_map()
        self.settings = self.world.get_settings()
        self.num_vehicles = num_vehicles
        self.num_peds = num_peds

        # Simulation traffic
        self.all_traffic = []

        # Carla commands
        self.spawn_actor = carla.command.SpawnActor
        self.auto_pilot = carla.command.SetAutopilot
        self.future_actor = carla.command.FutureActor

        # Spawn Traffic
        self._init_traffic_manager()
        if num_vehicles is not None:
            self.spawn_vehicles(num_vehicles)
        if num_peds is not None:
            self.spawn_walkers(num_peds)

    def _get_bp_lib(self, filter: str, generation: str) -> List:
        all_bps = self.world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return all_bps
        if len(all_bps) == 1:
            return all_bps
        
        try:
            gen = int(generation)
            if gen in [1, 2]:
                all_bps = [bp for bp in all_bps if int(bp.get_attribute("generation")) == gen]
                return all_bps
            else:
                print("Warning: Generation for traffic spawner is not valid.")
                return []
        except:
            print("Warning: Generation for traffic spawner is not valid.")
            return []
        
    def _init_traffic_manager(self, seed: int = None) -> None:
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if seed:
            self.traffic_manager.set_random_device_seed(seed)
        
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(30.0)

    def destroy_traffic(self) -> None:
        # Destroy vehicles
        if self.num_vehicles is not None:
            self.client.apply_batch([carla.command.DestroyActor(v) for v in self.all_vehicles])
        if self.num_peds is not None:
            for i in range(0, len(self.all_ids), 2):
                self.all_walker_actors[i].stop()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_ids])

        print("All traffic destroyed.")

    def spawn_walkers(self, num_walkers: int = 10, seed: int = 0) -> None:
        self.all_walkers = []
        self.all_ids = []
        walker_bps = self._get_bp_lib(filter="walker.pedestrian.*", generation="2")
        percent_peds_running = 0.0
        percent_peds_crossing = 0.0
        if seed:
            self.world.set_pedestrians_seed(seed)
            random.seed(seed)

        # Random spawn locations
        spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            location = self.world.get_random_location_from_navigation()
            if (location != None):
                spawn_point.location = location
                spawn_points.append(spawn_point)

        # Spawn walkers
        walkers_batch = []
        walkers_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            if walker_bp.has_attribute("speed"):
                if (random.random() > percent_peds_running):
                    walkers_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[1]
                    )
                else:
                    walkers_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[2]
                    )
            else:
                print("Warning: Walker has no speed.")
                walkers_speed.append(0.0)
            walkers_batch.append(self.spawn_actor(walker_bp, spawn_point))
        walkers_result = self.client.apply_batch_sync(walkers_batch, True)
        walkers_speed_ = []
        for i in range(len(walkers_result)):
            if walkers_result[i].error:
                print("Walker error:", walkers_result[i].error)
            else:
                self.all_walkers.append({"walker_id": walkers_result[i].actor_id})
                walkers_speed_.append(walkers_speed[i])
        walkers_speed = walkers_speed_

        # Spawn walkers controller
        controllers_batch = []
        walker_control_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        for i in range(len(self.all_walkers)):
            controllers_batch.append(self.spawn_actor(
                walker_control_bp, carla.Transform(), self.all_walkers[i]["walker_id"]
            ))
        controllers_result = self.client.apply_batch_sync(controllers_batch, True)
        for i in range(len(controllers_result)):
            if controllers_result[i].error:
                print("Walker controller error:", controllers_result[i].error)
            else:
                self.all_walkers[i]["controller"] = controllers_result[i].actor_id

        # Generate all IDs
        for i in range(len(self.all_walkers)):
            self.all_ids.append(self.all_walkers[i]["controller"])
            self.all_ids.append(self.all_walkers[i]["walker_id"])
        self.all_walker_actors = self.world.get_actors(self.all_ids)

        self.world.tick()
        self.world.set_pedestrians_cross_factor(percent_peds_crossing)
        for i in range(0, len(self.all_ids), 2):
            # Controller is at index i
            # Walker is at index i+1
            controller = self.all_walker_actors[i]
            walker = self.all_walker_actors[i+1]
            
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(float(walkers_speed[int(i/2)]))
            
            walker.set_collisions(True)
            walker.set_simulate_physics(True)

    def spawn_vehicles(self, num_vehicles: int = 10) -> None:
        self.all_vehicles = []

        vehicle_bps = self._get_bp_lib(filter="vehicle.*", generation="2")
        vehicle_bps = [v for v in vehicle_bps if int(v.get_attribute("number_of_wheels")) == 4
                       and v.id != "vehicle.miningtruck.miningtruck"]
        
        spawn_points = self.map.get_spawn_points()
        num_spawns = len(spawn_points)
        
        if num_vehicles < num_spawns:
            random.shuffle(spawn_points)
        elif num_vehicles > num_spawns:
            num_vehicles = num_spawns
            print("Vehicle spawn warning: Number of desired vehicles is greater than number of spawn points.")

        vehicles_batch = []
        for i, spawn_point in enumerate(spawn_points):
            if i >= num_vehicles:
                break
            vehicle_bp = np.random.choice(vehicle_bps)
            if vehicle_bp.has_attribute("color"):
                vehicle_color = np.random.choice(vehicle_bp.get_attribute("color").recommended_values)
                vehicle_bp.set_attribute("color", vehicle_color)
            if vehicle_bp.has_attribute("driver_id"):
                vehicle_id = np.random.choice(vehicle_bp.get_attribute("driver_id").recommended_values)
                vehicle_bp.set_attribute("driver_id", vehicle_id)
            vehicle_bp.set_attribute("role_name", "autopilot")

            vehicles_batch.append(
                self.spawn_actor(vehicle_bp, spawn_point).then(self.auto_pilot(
                    self.future_actor, True, self.traffic_manager.get_port()
                ))
            )

        for vehicles_result in self.client.apply_batch_sync(vehicles_batch, True):
            if vehicles_result.error:
                print("Vehicle spawn warning:", vehicles_result.error)
            else:
                self.all_vehicles.append(vehicles_result.actor_id)

        self.all_vehicle_actors = self.world.get_actors(self.all_vehicles)
        for vehicle in self.all_vehicle_actors:
            vehicle.set_collisions(True)
            vehicle.set_simulate_physics(True)
