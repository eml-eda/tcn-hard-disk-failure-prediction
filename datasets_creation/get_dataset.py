import os
import shutil
import wget
import zipfile

base_url = 'https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/'
script_dir = os.path.dirname(os.path.abspath(__file__))  # get the directory of the script
base_path = os.path.join(script_dir, '../../HDD_dataset/')  # build the relative path from the script directory
base_path = os.path.normpath(base_path)  # normalize the path (remove redundant separators and up-level references)
years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']

# zips contain different directory names or no directory at all, which causes
# unavoidable "spaghettiness" in the code
suffixes = {
        'data_2013.zip': '2013',
        'data_2014.zip': '2014',
        'data_2015.zip': '2015',
        'data_Q1_2016.zip': 'data_Q1_2016',
        'data_Q2_2016.zip': 'data_Q2_2016',
        'data_Q3_2016.zip': 'data_Q3_2016',
        'data_Q4_2016.zip': None,
        'data_Q1_2017.zip': None,
        'data_Q2_2017.zip': None,
        'data_Q3_2017.zip': None,
        'data_Q4_2017.zip': 'data_Q4_2017',
        'data_Q1_2018.zip': 'data_Q1_2018',
        'data_Q2_2018.zip': None,
        'data_Q3_2018.zip': None,
        'data_Q4_2018.zip': 'data_Q4_2018',
        'data_Q1_2019.zip': 'drive_stats_2019_Q1',
        'data_Q2_2019.zip': 'data_Q2_2019',
        'data_Q3_2019.zip': 'data_Q3_2019'}


def main(years, base_path):
    os.makedirs(base_path, exist_ok=True)
    # just in case they are passed as int
    years = [str(_) for _ in years]
    for y in years:
        print("Year:", y)
        year_path = os.path.join(base_path, y)
        os.makedirs(year_path, exist_ok=True)
        for zip_name, unzip_dir in suffixes.items():
            if y in zip_name:
                url = base_url + zip_name
                zip_path = os.path.join(base_path, zip_name)
                if not os.path.exists(zip_path):
                    print("Downloading:", url)
                    wget.download(url, out=base_path)
                print("\nUnzipping:", zip_path)
                dest_path = os.path.join(base_path, unzip_dir) if unzip_dir else year_path
                os.makedirs(dest_path, exist_ok=True)  # ensure dest_path exists
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(dest_path)

                if unzip_dir is not None and unzip_dir != y:
                    unzip_path = dest_path
                    for f in os.listdir(unzip_path):
                        print("Moving file:", f)
                        src_file_path = os.path.join(unzip_path, f)
                        dest_file_path = os.path.join(year_path, f)
                        if os.path.isdir(src_file_path):
                            # If f is a directory, move its contents to the destination directory
                            for subf in os.listdir(src_file_path):
                                dest_subdir = os.path.join(dest_file_path, subf)
                                os.makedirs(os.path.dirname(dest_subdir), exist_ok=True)  # ensure the destination directory exists
                                shutil.move(os.path.join(src_file_path, subf), dest_subdir)
                        else:
                            # If f is a file, move it to the destination directory
                            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)  # ensure the destination directory exists
                            shutil.move(src_file_path, dest_file_path)
                        print("Moved file:", f, "to:", dest_file_path)
                    print("Removing empty directory:", unzip_path)
                    os.rmdir(unzip_path)



if __name__ == "__main__":
    main(years, base_path)
