import os
import numpy as np
import pandas as pd
import datetime
from config import *

# Load the hard drives data
hdd_model_data = set(np.load(hdd_model_file_path))

database = pd.DataFrame()

# Iterate over each year
for year in years:
    year_path = year_dirs[year]
    files = sorted([f for f in os.listdir(year_path) if f.endswith('.csv')])

    # Iterate over each file in the directory
    for file in files:
        file_path = os.path.join(year_path, file)
        file_date = datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')
        old_time = datetime.datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
        
        if file_date >= old_time:
            df = pd.read_csv(file_path)

            for model in models:
                model_chosen = df[df['model'] == model]
                relevant_rows = model_chosen[model_chosen['serial_number'].isin(hdd_model_data)]

                # Drop unnecessary columns since the following columns are not standard for all models
                drop_columns = [col for col in relevant_rows if 'smart_' in col and int(col.split('_')[1]) in {22, 220, 222, 224, 226}]
                relevant_rows.drop(columns=drop_columns, errors='ignore', inplace=True)

                # Append the row to the database
                database = pd.concat([database, relevant_rows], ignore_index=True)
                print('adding day ' + str(model_chosen['date'].values))

# Save the database to a pickle file
database.to_pickle(hdd_model_pkl_file_path)