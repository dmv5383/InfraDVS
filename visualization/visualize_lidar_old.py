import open3d as o3d
import os
import time
import numpy as np
import ast

point_clouds_directory = "Independent_Study/visualization/test_data/data/car_lidar_2/"

# Get a sorted list of all .pcd files in the directory
pcd_files = sorted([f for f in os.listdir(point_clouds_directory) if f.endswith('.pcd')])

def load_bounding_boxes(file_path):
    if not os.path.exists(file_path):
        print(f"Bounding box file {file_path} does not exist.")
        return []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        print(f"Parsing line: {line.strip()}")
        print(f"Parts: {parts}")
        obj_type = parts[0]
        try:
            # Parse the nested list values
            bbox = ast.literal_eval(parts[4])
            dimensions = [float(parts[8]), float(parts[9]), float(parts[10])]
            location = [float(parts[11]), float(parts[12]), float(parts[13])]
            rotation_y = float(parts[14])
        except ValueError as e:
            print(f"Error parsing line: {line.strip()}")
            print(f"Exception: {e}")
            continue

        # Create a bounding box following KITTI format
        center = np.array(location)
        extent = np.array(dimensions)
        R = o3d.geometry.get_rotation_matrix_from_xyz([0, rotation_y, 0])
        box = o3d.geometry.OrientedBoundingBox(center, R, extent)
        box.color = (1, 0, 0)  # Red color for bounding box
        boxes.append(box)
    print(f"Loaded {len(boxes)} bounding boxes from {file_path}")
    return boxes

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Load the first point cloud to initialize the geometry
pcd_path = os.path.join(point_clouds_directory, pcd_files[0])
pcd = o3d.io.read_point_cloud(pcd_path)
vis.add_geometry(pcd)

# Set the view to isometric
view_control = vis.get_view_control()
view_control.set_front([0.5, -0.5, -0.5])
view_control.set_lookat([0, 0, 0])
view_control.set_up([0, 1, 0])
view_control.set_zoom(0.8)

# Loop through each file and update the geometry
for pcd_file in pcd_files:
    pcd_path = os.path.join(point_clouds_directory, pcd_file)
    pcd.points = o3d.io.read_point_cloud(pcd_path).points

    # Extract frame number from the pcd file name
    frame_number = os.path.splitext(pcd_file)[0].split('_')[0]

    # Remove previous geometries
    vis.clear_geometries()
    vis.add_geometry(pcd)

    # Add bounding boxes if the file exists
    bbox_file = f"{frame_number}_bboxes.txt"
    bbox_path = os.path.join(point_clouds_directory, bbox_file)
    print(f"Looking for bounding box file: {bbox_path}")
    boxes = load_bounding_boxes(bbox_path)
    for box in boxes:
        vis.add_geometry(box)
        print(f"Added bounding box: {box}")

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # Adjust the delay as needed

vis.destroy_window()