import shutil
import os
shutil.rmtree('celeba_results_images')
os.mkdir('celeba_results_images')


for file_ in os.listdir('training_artifacts/celeba'):
    if ('.jpg' in file_ or '.png' in file_) and '_64_' in file_:
        shutil.copy(f'training_artifacts/celeba/{file_}',f'celeba_results_images/{file_}')

shutil.make_archive('celeba_results_download','zip','celeba_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo/ALAE/celeba_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
