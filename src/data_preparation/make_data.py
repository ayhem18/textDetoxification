"""
This script contains functionalities to retrieve the data and make the different datasets used across the project
"""

import os, sys
from pathlib import Path
import requests as r


HOME = os.getcwd()
current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'data_analysis'))
sys.path.append(os.path.join(str(current), 'evaluation'))
sys.path.append(os.path.join(str(current), 'text_processing'))

from src.training_utilities.directories_and_files import unzip_data_file, squeeze_directory
import src.text_processing.preprocess as pr
import src.data_preparation.prepare_data as prd

import shutil

import requests
import zipfile
from io import BytesIO
import os


def download_and_unzip_google_drive_zip(public_link, output_directory):
    # Extract the file ID from the public link
    file_id = public_link.split("/")[-2]

    # Construct the download link for the file
    download_url = f"https://drive.google.com/uc?id={file_id}"

    try:
        # Send a GET request to the download link
        response = requests.get(download_url)
        response.raise_for_status()

        # Create a BytesIO object from the response content
        zip_data = BytesIO(response.content)

        # Unzip the file to the specified output directory
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(output_directory)

        print(f"File downloaded and unzipped to {output_directory}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


# def download_google_drive_file(public_link, path):
#     # Extract the file ID from the public link
#     file_id = public_link.split("/")[-2]

#     # Construct the download link for the file
#     download_url = f"https://drive.google.com/uc?id={file_id}"

#     try:
#         # Send a GET request to the download link
#         response = requests.get(download_url, allow_redirects=True)
#         response.raise_for_status()

#         with open(os.path.join(path), 'wb') as file:
#             file.write(response.content)
#         file.close()
            
#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")


# def download_initial_data():
#     response = r.get(_filtered_dataset_link, allow_redirects=True)
#     data_folder = os.path.join(PARENT_DIR, 'data')
#     os.makedirs(data_folder, exist_ok=True)
#     data_zip_file = os.path.join(PARENT_DIR, 'data', 'initial.zip')

#     with open(data_zip_file, mode="wb") as file:
#         file.write(response.content)

#     # unzip the file
#     filtered =  unzip_data_file(data_zip_file, 
#                     unzip_directory=os.path.join(PARENT_DIR, 'data'))

#     # extract the csv file from the generated folder
#     shutil.move(os.path.join(data_folder, 'initial', 'filtered.tsv'), os.path.join(data_folder, 'filtered.tsv'))
#     shutil.rmtree(os.path.join(data_folder, 'initial'))
#     os.remove(os.path.join(data_folder, 'initial.zip'))

#     # download the checkpoints
#     url = "https://drive.google.com/file/d/1J2Mz5u7hKyNYSJoaj1Hfp18X5qmhGqcJ/view?usp=drive_link"
#     output = os.path.join(PARENT_DIR, 'checkpoints.zip')
#     gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
#     unzip_data_file(output, unzip_directory=PARENT_DIR)
#     os.remove(output)


#     url

def download_checkpoints():
    # download checkpoints
    url = "https://drive.google.com/file/d/1J2Mz5u7hKyNYSJoaj1Hfp18X5qmhGqcJ/view?usp=drive_link"
    output = os.path.join(PARENT_DIR, 'checkpoints.zip')
    gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
    unzip_data_file(output, unzip_directory=os.path.join(PARENT_DIR, 'checkpoints'))
    os.remove(output)
    squeeze_directory(os.path.join(PARENT_DIR, 'checkpoints'))

def download_data():
    # download the data
    url = "https://drive.google.com/file/d/1MM6--rdUht-bsKN-0TG5ZNmBcXm1p-Y8/view?usp=sharing"
    output = os.path.join(PARENT_DIR, 'data.zip')
    gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
    unzip_data_file(output, unzip_directory=os.path.join(PARENT_DIR))
    os.remove(output)
    squeeze_directory(os.path.join(PARENT_DIR, 'data'))


import gdown
if __name__ == '__main__':
    path=os.path.join(PARENT_DIR, 'data')
    # download_and_unzip_google_drive_zip(_my_data_link, output_directory=path)
    ready = False
    while not ready:
        try: 
            download_data()
        except:
            continue
        ready = True

    ready = False

    while not ready:
        try: 
            print("trying download")
            download_checkpoints()            
        except:
            continue
        ready = True
    
        

