import os

base_url = 'https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/'
model = 'ST3000DM001'

# zips contain different directory names or no directory at all, which causes
# unavoidable "spaghettiness" in the code
suffixes = {
    'data_2013.zip': '2013',
    'data_2014.zip': '2014',
    'data_2015.zip': '2015',
    'data_Q1_2016.zip': None,
    'data_Q2_2016.zip': None,
    'data_Q3_2016.zip': None,
    'data_Q4_2016.zip': None,
    'data_Q1_2017.zip': None,
    'data_Q2_2017.zip': None,
    'data_Q3_2017.zip': None,
    'data_Q4_2017.zip': None,
    'data_Q1_2018.zip': None,
    'data_Q2_2018.zip': None,
    'data_Q3_2018.zip': None,
    'data_Q4_2018.zip': None,
    'data_Q1_2019.zip': None,
    'data_Q2_2019.zip': None,
    'data_Q3_2019.zip': None
}

failed = False  # This should be set based on your specific criteria or kept as a placeholder
years = [str(year) for year in range(2013, 2020)]

# Define the directory of HDD_dataset, it is inside project folder, and now parallel with the 'algorithms' and 'datasets_creation' folders
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.normpath(os.path.join(script_dir, '..', 'HDD_dataset'))

# Define the directories for each year
year_dirs = {year: os.path.join(base_path, year) for year in years}

years_list = "_" + "_".join(years)
suffix = 'failed' if failed else 'all'

# Define generated files directory
output_dir = os.path.join(script_dir, '..', 'output')

# Ensure the directory exists before attempting to save
os.makedirs(output_dir, exist_ok=True)

# Define generated file paths
hdd_model_file_path = os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model}.npy')
hdd_model_pkl_file_path = os.path.join(script_dir, '..', 'output', f'HDD{years_list}_{suffix}_{model}_appended.pkl')

# Define the final generated pkl file path
hdd_model_final_pkl_file_path = os.path.join(output_dir, f'{model}{years_list}_{suffix}.pkl')
