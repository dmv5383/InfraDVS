import os
import argparse
import numpy as np
import open3d as o3d
import math
import glob
import time
import threading
import cv2  # Add OpenCV for video recording

def parse_kitti_format(bbox_file):
    """Parse KITTI format bounding boxes from a text file."""
    bboxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the KittiDescriptor string format
            try:
                # Example format: "car,0,0,-10, ,2.560655355453491 1.9884328842163086 5.91519021987915,-3.4782168865203857 4.501531958580017 3.6491470336914062,-3.1412954674386064,55"
                parts = line.split(',')
                if len(parts) < 9:  # Basic check for valid format
                    continue
                
                obj_type = parts[0]
                truncated = float(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])
                bbox = parts[4]  # Ignored in this implementation
                dimensions = list(map(float, parts[5].split()))
                location = list(map(float, parts[6].split()))
                rotation_y = float(parts[7])
                obj_id = int(parts[8])
                
                # Convert from KITTI camera to CARLA/Open3D coordinates
                # Properly handle z-axis flipping based on KITTI convention
                y = location[0]
                z = -location[1]
                x = location[2]  # Negate the already negated z to get back to CARLA coords
                
                location = [x, y, z]
                
                bboxes.append({
                    'type': obj_type,
                    'dimensions': dimensions,  # height, width, length
                    'location': location,      # x, y, z in CARLA coords (ground-level, not center)
                    'rotation_y': rotation_y,
                    'id': obj_id
                })
                
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Exception: {e}")
                continue
                
    return bboxes

def create_bbox_from_kitti(bbox_data):
    """Create Open3D bounding box from KITTI format data."""
    # In the bbox_data, dimensions are [height, width, length]
    height, width, length = bbox_data['dimensions']
    x, y, z = bbox_data['location']
    rotation_y = bbox_data['rotation_y']
    
    # KITTI format puts the box center on the ground - adjust z by adding half the height
    # to get the true center of the bounding box
    z_adjusted = z + height/2
    
    # Create oriented bounding box
    # In Open3D, extent is the full size (not half-size like in CARLA)
    box = o3d.geometry.OrientedBoundingBox(
        center=[x, y, z_adjusted],  # Use adjusted z-coordinate
        R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, rotation_y]),
        extent=[length, width, height]  # length, width, height to match Open3D convention
    )
    
    # Assign color based on object type
    if bbox_data['type'].lower() == 'car':
        box.color = np.array([1, 0, 0])  # Red for cars
    elif bbox_data['type'].lower() == 'pedestrian':
        box.color = np.array([0, 1, 0])  # Green for pedestrians
    else:
        box.color = np.array([0, 0, 1])  # Blue for others
        
    return box

def load_point_cloud(pcd_file):
    """Load point cloud from PCD file."""
    print(f"Loading point cloud from: {pcd_file}")
    try:
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        # Convert to legacy PointCloud for visualization compatibility
        legacy_pcd = o3d.geometry.PointCloud()
        legacy_pcd.points = o3d.utility.Vector3dVector(pcd.point["positions"].numpy())
        
        # If intensity information is available, use it for coloring
        if "intensity" in pcd.point:
            intensities = pcd.point["intensity"].numpy()
            # Normalize intensities to [0, 1] for coloring
            if intensities.size > 0:
                normalized = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-10)
                colors = np.zeros((normalized.shape[0], 3))
                # Create grayscale colors based on intensity
                colors[:, 0] = normalized.flatten()  # R
                colors[:, 1] = normalized.flatten()  # G
                colors[:, 2] = normalized.flatten()  # B
                legacy_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"Point cloud loaded with {len(legacy_pcd.points)} points")
        return legacy_pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None

def visualize_point_cloud_with_bboxes(pcd_file, bbox_file):
    """Visualize point cloud with 3D bounding boxes."""
    # Load point cloud
    pcd = load_point_cloud(pcd_file)
    if pcd is None:
        return
    
    # Parse bounding boxes
    bboxes_data = parse_kitti_format(bbox_file)
    print(f"Found {len(bboxes_data)} bounding boxes")
    
    # Create visualization geometries
    vis_geometries = [pcd]
    
    # Create coordinate frame for reference (size=2 meters)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis_geometries.append(coord_frame)
    
    # Create and add bounding boxes to visualization
    for bbox_data in bboxes_data:
        try:
            bbox = create_bbox_from_kitti(bbox_data)
            vis_geometries.append(bbox)
            
            # Add a small sphere at the location of the bounding box center for better visualization
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(bbox_data['location'])
            sphere.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
            vis_geometries.append(sphere)
            
            # Print info about this bounding box for debugging
            #print(f"Added bbox: Type={bbox_data['type']}, Location={bbox_data['location']}, "
            #      f"Dimensions={bbox_data['dimensions']}, Rotation={bbox_data['rotation_y']}")
            
        except Exception as e:
            print(f"Error creating bounding box: {e}")
    
    # Visualize
    print("Visualizing point cloud with bounding boxes...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"LiDAR Point Cloud with Bounding Boxes - {os.path.basename(pcd_file)}",
        width=1560,
        height=1440
    )
    
    for geom in vis_geometries:
        vis.add_geometry(geom)
    
    # Set initial view control parameters
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 2.0  # Larger points
    
    # Set the initial camera viewpoint to look from above
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # Looking down the negative z-axis (from above)
    view_control.set_up([0, 1, 0])      # Y is up (changed from negative to positive)
    view_control.set_lookat([0, 0, 0])  # Look at the center
    view_control.set_zoom(0.5)          # Adjust zoom as needed
    
    vis.run()
    vis.destroy_window()

def find_matching_files(data_dir, frame_number=None):
    """Find matching point cloud and bounding box files."""
    pairs = []
    
    if frame_number is not None:
        # Look for specific frame
        frame_str = f"{int(frame_number):08d}"
        pcd_file = os.path.join(data_dir, f"{frame_str}_cloud.pcd")
        bbox_file = os.path.join(data_dir, f"{frame_str}_bboxes.txt")
        
        if os.path.exists(pcd_file) and os.path.exists(bbox_file):
            pairs.append((pcd_file, bbox_file))
    else:
        # Find all matching pairs
        pcd_files = sorted(glob.glob(os.path.join(data_dir, "*_cloud.pcd")))
        
        for pcd_file in pcd_files:
            frame_str = os.path.basename(pcd_file).split("_")[0]
            bbox_file = os.path.join(data_dir, f"{frame_str}_bboxes.txt")
            
            if os.path.exists(bbox_file):
                pairs.append((pcd_file, bbox_file))
    
    return pairs

def visualize_sequence(file_pairs, output_video=None):
    """Visualize a sequence of point clouds with bounding boxes and optionally save as a video."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR Point Cloud Sequence", width=1560, height=1440)
    
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, 10.0, (1560, 1440))
    else:
        video_writer = None
        print(
            "--------- Mouse view control ----------\n"
            "Left button + drag         : Rotate.\n"
            "Ctrl + left button + drag  : Translate.\n"
            "Wheel button + drag        : Translate.\n"
            "Shift + left button + drag : Roll.\n"
            "Wheel                      : Zoom in/out.\n"
        )
    
    def capture_frame(vis):
        if video_writer:
            image = vis.capture_screen_float_buffer(False)
            image = np.asarray(image)
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)
    
    def update_visualization(vis, index):
        vis.clear_geometries()
        pcd_file, bbox_file = file_pairs[index]
        pcd = load_point_cloud(pcd_file)
        bboxes_data = parse_kitti_format(bbox_file)
        
        vis.add_geometry(pcd)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(coord_frame)
        
        for bbox_data in bboxes_data:
            bbox = create_bbox_from_kitti(bbox_data)
            vis.add_geometry(bbox)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(bbox_data['location'])
            sphere.paint_uniform_color([1.0, 0.5, 0.0])
            vis.add_geometry(sphere)
        
        capture_frame(vis)
        vis.poll_events()
        vis.update_renderer()
    
    current_index = 0
    update_visualization(vis, current_index)
    
    def next_frame(vis):
        nonlocal current_index
        if current_index < len(file_pairs) - 1:
            current_index += 1
            update_visualization(vis, current_index)
    
    def prev_frame(vis):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_visualization(vis, current_index)
    
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    
    # Set initial view control parameters
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 2.0  # Larger points
    
    if output_video:
        while current_index < len(file_pairs) - 1:
            next_frame(vis)
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()
    
    if video_writer:
        video_writer.release()

def main():
    parser = argparse.ArgumentParser(description='Visualize LiDAR point clouds with bounding boxes.')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing point clouds and bounding boxes')
    parser.add_argument('--frame', type=int, default=None,
                        help='Specific frame number to visualize (default: visualize all frames)')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Run visualization in interactive mode (opens window for each frame)')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Output video file to save the sequence visualization')
    
    args = parser.parse_args()
    
    file_pairs = find_matching_files(args.data_dir, args.frame)
    
    if not file_pairs:
        print(f"No matching point cloud and bounding box files found in {args.data_dir}")
        if args.frame is not None:
            print(f"for frame {args.frame}")
        return
    
    print(f"Found {len(file_pairs)} point cloud/bounding box pairs")
    
    if args.output_video or not args.interactive:
        visualize_sequence(file_pairs, args.output_video)
    else:
        for pcd_file, bbox_file in file_pairs:
            print(f"\nVisualizing: {os.path.basename(pcd_file)}")
            visualize_point_cloud_with_bboxes(pcd_file, bbox_file)
            
            if not args.interactive and len(file_pairs) > 1:
                user_input = input("Press Enter for next frame or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break

if __name__ == "__main__":
    main()


"""
/Users/doryforde/miniforge3/envs/CARLA/bin/python /Users/doryforde/Desktop/School/Independent_Study/visualization/visualize_lidar.py --data_dir Independent_Study/visualization/test_data/data/Town10HD_ClearNoon_1/car_lidar --output_video Independent_Study/visualization/output/lidar_out.mp4
"""