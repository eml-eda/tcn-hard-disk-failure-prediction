import os
import pandas as pd
import numpy as np
# import datetime
from glob import glob

# Define paths and directorie
script_dir = os.path.dirname(os.path.abspath(__file__))
# Update the direcotry of HDD_dataset, it is inside project folder, and now parallel with the 'algorithms' and 'datasets_creation' folders
base_path = os.path.normpath(os.path.join(script_dir, '..', 'HDD_dataset'))
data_dir = os.path.dirname(base_path)

# Define the directories for each year
year_dirs = {str(year): os.path.join(base_path, str(year)) for year in range(2013, 2020)}

years = [str(year) for year in range(2013, 2018)]
model = 'ST3000DM001'
list_failed = []
failed = False

# Create temp directory
temp_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(temp_dir, exist_ok=True)

# Process each year
for year in years:
    year_dir = year_dirs[year]
    files = glob(os.path.join(year_dir, '*.csv'))

    # Process each file
    for file_path in sorted(files):
        try:
            file_r = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            continue

        # Filter the model with the chosen model
        model_chosen = file_r[file_r['model'] == model]
        #print(f"Number of entries after filtering by model: {len(model_chosen)}")
        
        # Print processing day
        print('processing day ' + str(model_chosen['date'].values))

        if failed:
            # Filter the failed hard drives
            model_chosen = model_chosen[model_chosen['failure'] == 1]
            print(f"Number of entries after filtering by failure: {len(model_chosen)}")

        # Append serial numbers
        list_failed.extend(model_chosen['serial_number'].values)

# Save the list of failed or all hard drives
suffix = 'failed' if failed else 'all'
np.save(os.path.join(temp_dir, f'HDD_{suffix}_{model}'), list_failed)