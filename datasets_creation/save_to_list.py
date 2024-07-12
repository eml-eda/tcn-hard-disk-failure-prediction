import os
import pandas as pd
import numpy as np
from glob import glob


def save_to_list(model, years, failed, base_path, output_dir):
    list_failed = []
    years_list = "_" + "_".join(years)
    models = [m.strip() for m in model.split(',')]
    model_str = "_".join(models)
    suffix = 'failed' if failed else 'all'

    # Ensure the directory exists before attempting to save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    hdd_model_file_path = os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model_str}.npy')

    # Define the directories for each year
    year_dirs = {year: os.path.join(base_path, year) for year in years}

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
    return f'Model saved to {hdd_model_file_path}'