import os
import pandas as pd
from config import *

# Load the database
# Fix the loading of the appended database from 'output' directory
database = pd.read_pickle(hdd_model_pkl_file_path)

# Create grouped object once
grouped = database.groupby('serial_number')

# Initial base DataFrame using the 'date' column
base = grouped['date'].apply(list).to_frame()

# Efficient way to concatenate all features by using loop through remaining columns
for i, smart in enumerate(database.columns[4:], start=1):
    print(f'concatenating feature {i}/{len(database.columns[4:])}')
    base[smart] = grouped[smart].apply(list)

# Save the DataFrame
# Rename dataset to [model]_[years]_[suffix].pkl 
base.to_pickle(hdd_model_final_pkl_file_path)
