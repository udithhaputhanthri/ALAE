import shutil
import os
try:shutil.rmtree('citescapes_results_images')
except:pass
os.mkdir('citescapes_results_images')


for file_ in os.listdir('training_artifacts/citescapes'):
    if ('.jpg' in file_ or '.png' in file_) and '_4' in file_:
        shutil.copy(f'training_artifacts/citescapes/{file_}',f'citescapes_results_images/{file_}')

shutil.make_archive('citescapes_results_download','zip','citescapes_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/citescapes_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
