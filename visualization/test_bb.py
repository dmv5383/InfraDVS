import open3d as o3d
import os
import time
import numpy as np

point_clouds_directory = "Independent_Study/visualization/test_data/data/Town10HD_CloudyNoon_3/car_lidar/"

# Get a sorted list of all .pcd files in the directory
pcd_files = sorted([f for f in os.listdir(point_clouds_directory) if f.endswith('.pcd')])

def load_bounding_boxes(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        obj_type = parts[0]
        bbox = list(map(float, parts[4:8]))
        dimensions = list(map(float, parts[8:11]))
        location = list(map(float, parts[11:14]))
        rotation_y = float(parts[14])

        # Create a bounding box
        center = np.array(location)
        extent = np.array(dimensions)
        R = o3d.geometry.get_rotation_matrix_from_xyz([0, rotation_y, 0])
        box = o3d.geometry.OrientedBoundingBox(center, R, extent)
        box.color = (1, 0, 0)  # Red color for bounding box
        boxes.append(box)
    return boxes

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Load the first point cloud to initialize the geometry
pcd_path = os.path.join(point_clouds_directory, pcd_files[0])
pcd = o3d.io.read_point_cloud(pcd_path)
#vis.add_geometry(pcd)

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
    frame_number = os.path.splitext(pcd_file)[0].split('_')[-1]

    # Remove previous geometries
    vis.clear_geometries()
    #vis.add_geometry(pcd)

    # Add bounding boxes if the file exists
    bbox_file = f"bounding_boxes_{frame_number}.txt"
    bbox_path = os.path.join(point_clouds_directory, bbox_file)
    boxes = load_bounding_boxes(bbox_path)
    for box in boxes:
        vis.add_geometry(box)
        print(box)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # Adjust the delay as needed

vis.destroy_window()