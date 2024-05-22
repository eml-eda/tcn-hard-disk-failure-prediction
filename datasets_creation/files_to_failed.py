import os
import numpy as np
import pandas as pd
import datetime

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Fix relative path required for the dataset
base_path = os.path.abspath(os.path.join(script_dir, '..', 'HDD_dataset'))

# Define the directories for each year
year_dirs = {year: os.path.join(base_path, year) for year in map(str, range(2013, 2020))}

years = [str(year) for year in range(2013, 2018)]
model = 'ST3000DM001'

# Load the failed hard drives
# Fix the reading of the failed hard drives' path to 'output' directory
# failed = set(np.load(os.path.join(script_dir, '..', 'output', f'HDD_all_{model}.npy')))

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
            model_chosen = df[df['model'] == model]
            relevant_rows = model_chosen[model_chosen['serial_number'].isin(failed)]

            # Drop unnecessary columns since the following columns are not standard for all models
            drop_columns = [col for col in relevant_rows if 'smart_' in col and int(col.split('_')[1]) in {22, 220, 222, 224, 226}]
            relevant_rows.drop(columns=drop_columns, errors='ignore', inplace=True)

            # Append the row to the database
            database = pd.concat([database, relevant_rows], ignore_index=True)
            print('adding day ' + str(model_chosen['date'].values))

# Save the database to a pickle file
# Fix the saving of the appended database to 'output' directory
database.to_pickle(os.path.join(script_dir, '..', 'output', f'All_failed_appended_{model}.pkl'))