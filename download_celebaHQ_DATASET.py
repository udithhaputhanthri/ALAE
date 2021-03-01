
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

gdd.download_file_from_google_drive(file_id='1-LFFkFKNuyBO1sjkM4t_AArIXr3JAOyl',
                                    dest_path='/home/udith/data/datasets/celeba-hq/images.zip',
                                    unzip=True)

os.remove('/home/udith/data/datasets/celeba-hq/images.zip')
