import os
import csv
import numpy as np
import argparse
# Parse argument for path input
parser = argparse.ArgumentParser(description='Process CSV file path.')
parser.add_argument('--path', type=str, help='Path to the directory containing the CSV file.')
args = parser.parse_args()

# Use provided path or default to current directory
if args.path:
    csv_path = args.path
else:
    # Get the current directory
    current_directory = os.path.dirname(__file__)

    # Find the first CSV file in the directory
    csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]
    first_csv_file = csv_files[0] if csv_files else None

    if first_csv_file:
        csv_path = os.path.join(current_directory, first_csv_file)
    else:
        print("No CSV file found.")

        
with open(csv_path, mode='r') as file:
    reader = csv.reader(file)
    data_as_list = list(reader)

total_processing_time = []
mfcc_conversion_time = []
prediction_time = []
for row in data_as_list:
    if len(row) < 3:
        continue
    mfcc_conversion_time.append(float(row[-3]))
    prediction_time.append(float(row[-2]))
    total_processing_time.append( float(row[-1]))

print(f"Total processing time: {sum(total_processing_time)}s")
print(f"Average processing time: {sum(total_processing_time)/len(total_processing_time)}s, std: {np.std(total_processing_time)}")
print(f"Average MFCC conversion time: {sum(mfcc_conversion_time)/len(mfcc_conversion_time)}s, std: {np.std(mfcc_conversion_time)}")
print(f"Average prediction_time time: {sum(prediction_time)/len(prediction_time)}s, std: {np.std(prediction_time)}")