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
failed = np.load('../temp/HDD_all_'+ model + '.npy')
#failed = np.load('../temp/HDD_failed.npy')
failed = dict.fromkeys(failed).keys()
database = pd.DataFrame()
for year in years:
    old_time = datetime.datetime.strptime(year+'-01-01', '%Y-%m-%d')
    read_dir = os.path.join(data_dir,year_dir[year])
    for file in sorted(os.listdir(read_dir)):
        if os.path.isfile(os.path.join(read_dir,file)):
            if 'csv' in file:
                if datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')>=old_time:
                    file_r = pd.read_csv(os.path.join(read_dir,file))
                    model_chosen = file_r[file_r['model']==model]
                    row = model_chosen[model_chosen['serial_number'].isin(failed)]
                    if 'smart_22_raw' in row.columns:
                        row = row.drop(['smart_22_raw','smart_22_normalized','smart_220_raw','smart_220_normalized','smart_222_raw','smart_222_normalized','smart_224_raw','smart_224_normalized','smart_226_raw','smart_226_normalized'],axis=1)
                    database = database.append(row)
                    print('adding day ' + str(np.asarray(model_chosen['date'].values)))
database.to_pickle('../temp/All_failed_appended_' + model +'.pkl')
