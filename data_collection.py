import subprocess

# Get name of Docker container
def get_docker_container_name():
    result = subprocess.check_output("sudo docker ps --format '{{.Names}}'", shell=True).decode().strip()
    return result

# Create scenario
def create_scenario(map_name, world_weather, run_number, recording_length, num_vehicles, num_pedestrians):
    activate_conda_env()
    subprocess.run([
        "python3", "./create_scenario.py", 
        "--world_map", map_name, 
        "--world_weather", world_weather, 
        "--run_number", str(run_number), 
        "--record_start_time", "10", 
        "--record_delta_time", str(recording_length), 
        "--num_vehicles", str(num_vehicles), 
        "--num_peds", str(num_pedestrians), 
        "--out_path", "datasets/data/"
    ], shell=True)

# Copy datasets out of Docker container
def copy_datasets_out(container_name):
    subprocess.run(f"sudo docker cp {container_name}:/home/carla/datasets/ /data/CARLA_DVS_Scripts/InfraDVS/", shell=True)

# Restart Docker container
def restart_docker_container(container_name):
    subprocess.run(f"sudo docker container restart {container_name}", shell=True)

# Read scenario
def read_scenario(config_path, tick_rate, record_path, out_path, start_time):
    subprocess.run([
        "python3", "./read_scenario.py", 
        config_path,
        f"--tick_rate={tick_rate}",
        f"--record_path={record_path}",
        f"--out_path={out_path}",
        f"--start_time={start_time}"
    ], shell=True)

# Kill Docker container
def kill_docker_container(container_name):
    subprocess.run(f"sudo docker container kill {container_name}", shell=True)

# Usage Example
if __name__ == "__main__":
    # Get Docker container name
    container_name = get_docker_container_name()
    print(f"Container Name: {container_name}")

    # Example parameters for scenario creation
    create_scenario("Town03", "ClearNoon", 1, 10, 50, 30)
    
    # Copy datasets
    copy_datasets_out(container_name)
    
    # Restart Docker container
    restart_docker_container(container_name)
    
    # Read scenario (modify parameters as needed)
    read_scenario("camera_sensors.json", 0.001, "/home/carla/datasets/data/", "/output/data", 1.0)
    
    # Kill the Docker container
    kill_docker_container(container_name)