import shutil
import os
try:shutil.rmtree('celebahq-l2_results_images')
except:pass
os.mkdir('celebahq-l2_results_images')


for file_ in os.listdir('training_artifacts/celeba-hq256-onlyL2/'):
    if ('.jpg' in file_ or '.png' in file_):
        shutil.copy(f'training_artifacts/celeba-hq256-onlyL2/{file_}',f'celebahq-l2_results_images/{file_}')

shutil.make_archive('celebahq-l2_results_download','zip','celebahq-l2_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/celebahq-l2_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
