#!/bin/bash

# filepath: /home/dmv5383/CARLA_DVS_Scripts/InfraDVS/gen_data.sh

# Function to create a scenario
create_scenario() {
    python3 ./create_scenario.py --world_map "$1" --world_weather "$2" --run_number "$3" --record_start_time "$4" --record_delta_time "$5" --num_vehicles "$6" --num_peds "$7"
    sleep 5
    sudo docker cp ${container_name}:/home/carla/datasets/scenarios /home/dmv5383/CARLA_DVS_Scripts/InfraDVS/datasets/
    sudo docker container restart ${container_name}
    sleep 5
}

# Function to read a scenario
read_scenario() {
    local tick_rate
    if [[ "$5" == "./gen_data/camera_sensors.json" ]]; then
        tick_rate=0.001
    else
        tick_rate=0.1
    fi
    python3 ./read_scenario.py --world_map "$1" --world_weather "$2" --record_delta_time "$3" --start_time "$4" --sensors_config "$5" --record_path "$6" --tick_rate "$tick_rate"
    sleep 5
    sudo docker container restart ${container_name}
    sleep 5
}

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CARLA2

# Read input file
input_file="$1"
if [ ! -f "$input_file" ]; then
    echo "Input file not found!"
    exit 1
fi

# Read Docker container name
container_name=$(sudo docker ps --format "{{.Names}}")

# Create necessary directories inside the Docker container
sudo docker exec ${container_name} mkdir -p /home/carla/datasets/scenarios
sudo docker exec ${container_name} chmod -R 777 /home/carla/datasets

# Initialize variables with default values
world_map=("Town10HD")
world_weather=("ClearNoon")
run_number=(1)
record_start_time=(10)
record_delta_time=(60)
num_vehicles=(20)
num_peds=(20)
start_time=(0.0)

# Read input file line by line
declare -A config
while IFS='=' read -r key value; do
    IFS=',' read -r -a values <<< "$value"
    config["$key"]="${values[@]}"
done < "$input_file"

# Extract values from config
world_map=(${config["world_map"]:-${world_map[@]}})
world_weather=(${config["world_weather"]:-${world_weather[@]}})
run_number=(${config["run_number"]:-${run_number[@]}})
record_start_time=(${config["record_start_time"]:-${record_start_time[@]}})
record_delta_time=(${config["record_delta_time"]:-${record_delta_time[@]}})
num_vehicles=(${config["num_vehicles"]:-${num_vehicles[@]}})
num_peds=(${config["num_peds"]:-${num_peds[@]}})
start_time=(${config["start_time"]:-${start_time[@]}})

# Iterate over all combinations of configurations
for wm in "${world_map[@]}"; do
    for ww in "${world_weather[@]}"; do
        for rn in "${run_number[@]}"; do
            for rst in "${record_start_time[@]}"; do
                for rdt in "${record_delta_time[@]}"; do
                    for nv in "${num_vehicles[@]}"; do
                        for np in "${num_peds[@]}"; do
                            for st in "${start_time[@]}"; do
                                # Run create_scenario
                                create_scenario "$wm" "$ww" "$rn" "$rst" "$rdt" "$nv" "$np"

                                # Construct the record path
                                record_path="/home/carla/datasets/scenarios/${wm}_${ww}_${rn}.log"

                                # Run read_scenario with camera_sensors.json
                                read_scenario "$wm" "$ww" "$rdt" "$st" "./gen_data/camera_sensors.json" "$record_path"

                                # Run read_scenario with lidar_sensors.json
                                read_scenario "$wm" "$ww" "$rdt" "$st" "./gen_data/lidar_sensors.json" "$record_path"
                            done
                        done
                    done
                done
            done
        done
    done
done