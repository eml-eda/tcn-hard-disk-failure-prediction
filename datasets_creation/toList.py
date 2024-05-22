import os
import pandas as pd

model = 'ST3000DM001'

# Define whether to process only failed data or all data
failed = False  # This should be set based on your specific criteria or kept as a placeholder

# Load the database
# Fix the loading of the appended database from 'output' directory
database = pd.read_pickle(os.path.join('..', 'output', f'All_failed_appended_{model}.pkl'))

# Create grouped object once
grouped = database.groupby('serial_number')

# Initial base DataFrame using the 'date' column
base = grouped['date'].apply(list).to_frame()

# Efficient way to concatenate all features by using loop through remaining columns
for i, smart in enumerate(database.columns[4:], start=1):
    print(f'concatenating feature {i}/{len(database.columns[4:])}')
    base[smart] = grouped[smart].apply(list)

# Ensure the directory exists before attempting to save
# Fix the output directory path
output_dir = os.path.join('..', 'output')
os.makedirs(output_dir, exist_ok=True)

# Define the suffix based on the value of `failed`
suffix = 'failed' if failed else 'all'

# Save the DataFrame
# Rename dataset to [model]_[years]_[suffix].pkl 
base.to_pickle(os.path.join(output_dir, f'{model}_2013_2014_2015_2016_2017_{suffix}.pkl'))
