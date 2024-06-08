import os
import pandas as pd
import numpy as np
from glob import glob
from config import *

list_failed = []

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

        # Iterate over each model in the list
        for model in models:
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
np.save(hdd_model_file_path, list_failed)