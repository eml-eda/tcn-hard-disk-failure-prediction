import os
import pandas as pd


def save_to_grouped_list(model, years, failed, output_dir):
    years_list = "_" + "_".join(years)
    suffix = 'failed' if failed else 'all'
    models = [m.strip() for m in model.split(',')]
    model_str = "_".join(models)

    hdd_model_pkl_file_path = os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model_str}_appended.pkl')
    # Load the database
    database = pd.read_pickle(hdd_model_pkl_file_path)

    # Create grouped object once
    grouped = database.groupby('serial_number')

    # Initial base DataFrame using the 'date' column
    base = grouped['date'].apply(list).to_frame()

    # Efficient way to concatenate all features by using loop through remaining columns
    for i, smart in enumerate(database.columns[4:], start=1):
        print(f'concatenating feature {i}/{len(database.columns[4:])}')
        base[smart] = grouped[smart].apply(list)

    hdd_model_final_pkl_file_path = os.path.join(output_dir, f'{model_str}{years_list}_{suffix}.pkl')
    # Save the DataFrame
    base.to_pickle(hdd_model_final_pkl_file_path)
    return f'Model saved to {hdd_model_final_pkl_file_path}'