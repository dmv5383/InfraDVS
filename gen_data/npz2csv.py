import numpy as np
import csv
import sys
import os
import glob

def process_npz_directory_to_single_csv(input_dir, output_csv_path):
    """
    Processes all .npz files in the input directory and combines their data
    into a single CSV file with columns [x, y, polarity, timestamp],
    preserving the original order of files. No header is written.
    
    Args:
        input_dir (str): Directory containing .npz files
        output_csv_path (str): Path to the output .csv file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all .npz files in the input directory
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return False
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    # Open the output CSV file with explicit settings to avoid headers
    with open(output_csv_path, 'w', newline='') as csvfile:
        # Create writer with explicit settings to avoid auto-formatting
        writer = csv.writer(csvfile)
        
        total_events = 0
        processed_files = 0
        
        # Process each .npz file in order
        for npz_file in npz_files:
            try:
                print(f"Processing {npz_file}...")
                
                # Load the .npz file
                with np.load(npz_file) as npz_data:
                    # Get the first (or only) array in the .npz file
                    if len(npz_data.files) == 0:
                        print(f"Warning: No arrays found in {npz_file}. Skipping.")
                        continue
                    
                    # This assumes all .npy files have the same name inside their respective .npz files
                    key = npz_data.files[0]
                    data = npz_data[key]
                    
                    # Check if data has the expected shape
                    if data.shape[1] != 4:
                        print(f"Warning: Expected 4 columns in {npz_file}, but found {data.shape[1]}. Skipping.")
                        continue
                    
                    # Write data with reordered columns without any headers
                    for row in data:
                        x, y, timestamp, polarity = row
                        # Only write numerical data, skip any potential string/header rows
                        if all(isinstance(val, (int, float, np.integer, np.floating)) for val in row):
                            writer.writerow([x, y, polarity, timestamp])
                    
                    file_events = len(data)
                    total_events += file_events
                    processed_files += 1
                    print(f"Added {file_events} events from {npz_file}")
                    
            except Exception as e:
                print(f"Error processing {npz_file}: {str(e)}")
    
    if processed_files > 0:
        print(f"Conversion complete. Successfully processed {processed_files} out of {len(npz_files)} files.")
        print(f"Total events written to {output_csv_path}: {total_events}")
        return True
    else:
        print("No data was successfully processed.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python npz2csv.py input_directory output.csv")
    else:
        input_dir = sys.argv[1]
        output_csv = sys.argv[2]
        process_npz_directory_to_single_csv(input_dir, output_csv)