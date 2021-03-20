import shutil
import os
try:shutil.rmtree('citescapes_high_epochs_results_images')
except:pass
os.mkdir('citescapes_high_epochs_results_images')


for file_ in os.listdir('training_artifacts/citescapes_high_epochs'):
    if ('.jpg' in file_ or '.png' in file_) and '4' in file_:
        shutil.copy(f'training_artifacts/citescapes_high_epochs/{file_}',f'citescapes_high_epochs_results_images/{file_}')

shutil.make_archive('citescapes_high_epochs_results_download','zip','citescapes_high_epochs_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/citescapes_high_epochs_results_download.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/')
