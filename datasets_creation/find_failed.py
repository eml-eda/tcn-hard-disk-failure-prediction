import os
import pandas as pd
import datetime
import numpy as np
data_dir = os.path.dirname('../../HDD_dataset/')
year_dir ={'2013': os.path.dirname('2013/'), '2014': os.path.dirname('2014/'),'2015': os.path.dirname('2015/'),\
           '2016': os.path.dirname('2016/'),'2017': os.path.dirname('2017/'), '2018': os.path.dirname('2018/'),\
           '2019': os.path.dirname('2019/')}
years = ['2013','2014','2015','2016','2017']
model = 'ST3000DM001'
list_failed = []
failed = True

# Create temp directory
temp_dir = os.path.join(script_dir, '..', 'temp')
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
            #print(f"Number of entries after filtering by failure: {len(model_chosen)}")

        # Append serial numbers
        list_failed.extend(model_chosen['serial_number'].values)

# Save the list of failed or all hard drives
suffix = 'failed' if failed else 'all'
np.save(os.path.join(temp_dir, f'HDD_{suffix}_{model}'), list_failed)
