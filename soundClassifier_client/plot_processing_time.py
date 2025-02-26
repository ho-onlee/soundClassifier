import os
import csv

# Get the current directory
current_directory = os.path.dirname(__file__)

# Find the first CSV file in the directory
csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]
first_csv_file = csv_files[0] if csv_files else None

if first_csv_file:
    csv_path = os.path.join(current_directory, first_csv_file)
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        data_as_list = list(reader)
else:
    print("No CSV file found.")