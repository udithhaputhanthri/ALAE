
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

celebahq256='1O89DVCoWsMhrIF3G8-wMOJ0h7LukmMdP'
celebahq1024= '1-LFFkFKNuyBO1sjkM4t_AArIXr3JAOyl'

gdd.download_file_from_google_drive(file_id=celebahq256,
                                    dest_path='/home/udith/data/datasets/celeba-hq/images.zip',
                                    unzip=True)

os.remove('/home/udith/data/datasets/celeba-hq/images.zip')
