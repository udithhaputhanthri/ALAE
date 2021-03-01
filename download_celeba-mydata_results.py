import shutil
import os
try:os.mkdir('celeba-mydata_results_images')
except:pass

for file_ in os.listdir('training_artifacts/celeba_mydata'):
    if '.jpg' in file_ or '.png' in file_:
        shutil.copy(f'training_artifacts/celeba_mydata/{file_}',f'celeba-mydata_results_images/{file_}')

shutil.make_archive('celeba-mydata_results_download','zip','celeba-mydata_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo/ALAE/celeba-mydata_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
