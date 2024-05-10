import os
import numpy as np
import pandas as pd
import datetime

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path from the script directory
base_path = os.path.join(script_dir, '../../HDD_dataset/')

# Normalize the path (remove redundant separators and up-level references)
base_path = os.path.normpath(base_path)

data_dir = os.path.dirname(base_path)

# Define the directories for each year
year_dir = {
    '2013': os.path.join(base_path, '2013/'),
    '2014': os.path.join(base_path, '2014/'),
    '2015': os.path.join(base_path, '2015/'),
    '2016': os.path.join(base_path, '2016/'),
    '2017': os.path.join(base_path, '2017/'),
    '2018': os.path.join(base_path, '2018/'),
    '2019': os.path.join(base_path, '2019/')
}

years = ['2013', '2014', '2015', '2016', '2017']
model = 'ST3000DM001'

# Load the failed hard drives
failed = np.load('../temp/HDD_all_'+ model + '.npy')
failed = dict.fromkeys(failed).keys()

database = pd.DataFrame()

# Iterate over each year
for year in years:
    old_time = datetime.datetime.strptime(year+'-01-01', '%Y-%m-%d')
    read_dir = os.path.join(data_dir,year_dir[year])
    
    # Iterate over each file in the directory
    for file in sorted(os.listdir(read_dir)):
        if os.path.isfile(os.path.join(read_dir,file)):
            if 'csv' in file:
                if datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')>=old_time:
                    file_r = pd.read_csv(os.path.join(read_dir,file))
                    model_chosen = file_r[file_r['model']==model]
                    row = model_chosen[model_chosen['serial_number'].isin(failed)]
                    
                    # Drop unnecessary columns
                    if 'smart_22_raw' in row.columns:
                        row = row.drop(['smart_22_raw','smart_22_normalized','smart_220_raw','smart_220_normalized','smart_222_raw','smart_222_normalized','smart_224_raw','smart_224_normalized','smart_226_raw','smart_226_normalized'],axis=1)
                    
                    # Append the row to the database
                    database = database.append(row)
                    print('adding day ' + str(np.asarray(model_chosen['date'].values)))

# Save the database to a pickle file
database.to_pickle('../temp/All_failed_appended_' + model +'.pkl')