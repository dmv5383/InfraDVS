import os
import glob
import numpy as np
import open3d as o3d
import argparse

NUM_PAIRS = 100  # Number of pairs to accumulate

def load_point_cloud(pcd_file):
    """Load point cloud from PCD file."""
    pcd = o3d.t.io.read_point_cloud(pcd_file)
    return pcd

def save_point_cloud(pcd, filename):
    """Save point cloud to PCD file."""
    o3d.t.io.write_point_cloud(filename, pcd, write_ascii=True)

def save_bboxes(bboxes, filename):
    """Save bounding boxes to a text file."""
    with open(filename, 'w') as f:
        for bbox in bboxes:
            f.write(bbox)

def find_matching_files(data_dir):
    """Find matching point cloud and bounding box files."""
    pairs = []
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "*_cloud.pcd")))
    for pcd_file in pcd_files:
        frame_str = os.path.basename(pcd_file).split("_")[0]
        bbox_file = os.path.join(data_dir, f"{frame_str}_bboxes.txt")
        if os.path.exists(bbox_file):
            pairs.append((pcd_file, bbox_file))
    return pairs

def accumulate_point_clouds(data_dir, output_dir):
    """Accumulate point clouds and bounding boxes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_pairs = find_matching_files(data_dir)
    num_files = len(file_pairs)
    
    for i in range(0, num_files, NUM_PAIRS):
        accumulated_points = []
        accumulated_intensities = []
        accumulated_bboxes = []
        
        for j in range(i, min(i + NUM_PAIRS, num_files)):
            pcd_file, bbox_file = file_pairs[j]
            pcd = load_point_cloud(pcd_file)
            points = pcd.point["positions"].numpy()
            intensities = pcd.point["intensity"].numpy()
            
            accumulated_points.append(points)
            accumulated_intensities.append(intensities)
            
            with open(bbox_file, 'r') as f:
                accumulated_bboxes.extend(f.readlines())
        
        accumulated_points = np.vstack(accumulated_points)
        accumulated_intensities = np.vstack(accumulated_intensities)
        
        accumulated_pcd = o3d.t.geometry.PointCloud()
        accumulated_pcd.point["positions"] = o3d.core.Tensor(accumulated_points, o3d.core.float32)
        accumulated_pcd.point["intensity"] = o3d.core.Tensor(accumulated_intensities, o3d.core.float32)
        
        output_pcd_file = os.path.join(output_dir, f"{i//NUM_PAIRS:08d}_cloud.pcd")
        output_bbox_file = os.path.join(output_dir, f"{i//NUM_PAIRS:08d}_bboxes.txt")
        
        save_point_cloud(accumulated_pcd, output_pcd_file)
        save_bboxes(accumulated_bboxes, output_bbox_file)
        print(f"Saved accumulated files: {output_pcd_file}, {output_bbox_file}")

def main():
    parser = argparse.ArgumentParser(description='Accumulate LiDAR point clouds and bounding boxes.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing point clouds and bounding boxes')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save accumulated point clouds and bounding boxes')
    
    args = parser.parse_args()
    
    accumulate_point_clouds(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()



"""
/Users/doryforde/miniforge3/envs/CARLA/bin/python /Users/doryforde/Desktop/School/Independent_Study/visualization/accumulate_point_clouds.py --data_dir visualization/test_data/data/Town10HD_ClearNoon_10/car_lidar --output_dir visualization/test_data/data/Town10HD_ClearNoon_10/car_lidar_acc
"""