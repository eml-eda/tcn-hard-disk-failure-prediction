import os
import shutil
import wget
import zipfile
from config import *

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
                dest_path = year_path if unzip_dir is None else base_path
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(dest_path)

                if unzip_dir is not None and unzip_dir != y:
                    unzip_path = os.path.join(dest_path, unzip_dir)
                    for f in os.listdir(unzip_path):
                        shutil.move(os.path.join(unzip_path, f),
                                os.path.join(year_path, f))
                    os.rmdir(unzip_path)

if __name__ == "__main__":
    main(years, base_path)
