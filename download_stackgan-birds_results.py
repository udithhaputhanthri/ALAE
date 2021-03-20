import shutil
import os
try:shutil.rmtree('stackgan-birds_results_images')
except:pass
os.mkdir('stackgan-birds_results_images')

i=0

results_len= len(os.listdir('training_artifacts/stackgan-birds'))
for file_ in os.listdir('training_artifacts/stackgan-birds'):
    if ('.jpg' in file_ or '.png' in file_):
        i+=1
        if results_len>600:
           if i%50!=0:continue
        shutil.copy(f'training_artifacts/stackgan-birds/{file_}',f'stackgan-birds_results_images/{file_}')

shutil.make_archive('stackgan-birds_results_download','zip','stackgan-birds_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/stackgan-birds_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
