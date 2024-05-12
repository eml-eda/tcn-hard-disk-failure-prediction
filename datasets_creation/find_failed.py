import os
import pandas as pd
import numpy as np
import datetime
from glob import glob

# Define paths and directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'HDD_dataset'))
data_dir = os.path.dirname(base_path)

# Define the directories for each year
year_dirs = {str(year): os.path.join(base_path, str(year)) for year in range(2013, 2020)}

years = [str(year) for year in range(2013, 2018)]
model = 'ST3000DM001'
list_failed = []
failed = False

# Create temp directory
temp_dir = os.path.join(script_dir, '..', 'temp')
os.makedirs(temp_dir, exist_ok=True)

# Process each year
for year in years:
    year_dir = year_dirs[year]
    files = glob(os.path.join(year_dir, '*.csv'))

    # Process each file
    for file_path in sorted(files):
        file_r = pd.read_csv(file_path)
        model_chosen = file_r[file_r['model'] == model]
        
        # Print processing day
        print('processing day ' + str(model_chosen['date'].values))

        if failed:
            model_chosen = model_chosen[model_chosen['failure'] == 1]

        # Append serial numbers
        list_failed.extend(model_chosen['serial_number'].values)

# Save the list of failed or all hard drives
suffix = 'failed' if failed else 'all'
np.save(os.path.join(temp_dir, f'HDD_{suffix}_{model}'), list_failed)