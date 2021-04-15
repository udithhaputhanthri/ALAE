artifact_dir = 'training_artifacts/celeba-hq256-loss_experiments'
output_dir = 'celeba-hq256-loss_experiments'

import shutil
import os
try:shutil.rmtree(output_dir)
except:pass
os.mkdir(output_dir)

for exp in os.listdir(artifact_dir):
    if exp!='exp13':continue
    try:os.mkdir(f'{output_dir}/{exp}')
    except:pass
    for file_ in os.listdir(f'{artifact_dir}/{exp}'):
        if ('.jpg' in file_ or '.png' in file_):
            shutil.copy(f'{artifact_dir}/{exp}/{file_}',f'{output_dir}/{exp}/{file_}')

shutil.make_archive(output_dir,'zip',output_dir)

print(f'Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/{output_dir}.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/tests/celeba-hq256-loss_experiments')
