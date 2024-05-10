import os
import pandas as pd
import datetime
import numpy as np

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
list_failed = []
failed = False

# Iterate over each year
for year in years:
    first = 1
    old_time = datetime.datetime.strptime(year+'-01-01', '%Y-%m-%d')
    read_dir = os.path.join(data_dir,year_dir[year])
    
    # Iterate over each file in the directory
    for file in sorted(os.listdir(read_dir)):
        if os.path.isfile(os.path.join(read_dir,file)):
            if 'csv' in file:
                if datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')>=old_time:
                    file_r = pd.read_csv(os.path.join(read_dir,file))
                    model_chosen = file_r[file_r['model']==model]
                    print('processing day ' + str(np.asarray(model_chosen['date'].values)))
                    if failed == True:
                        model_chosen = model_chosen[model_chosen['failure']==1]
                    for serial in model_chosen['serial_number'].values:
                        list_failed.append(serial)

# Create the directory if it doesn't exist
os.makedirs('../temp', exist_ok=True)

# Save the list of failed hard drives
if failed == True:
    np.save('../temp/HDD_failed_'+ model, list_failed)
else:
    np.save('../temp/HDD_all_'+ model, list_failed)