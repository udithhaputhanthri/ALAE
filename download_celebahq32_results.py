artifact_dir = 'training_artifacts/celeba-hq32/'
output_dir = 'celebahq32_results_images'

import shutil
import os
try:shutil.rmtree(output_dir)
except:pass
os.mkdir(output_dir)


for file_ in os.listdir(artifact_dir):
    if ('.jpg' in file_ or '.png' in file_):
        shutil.copy(f'{artifact_dir}/{file_}',f'{output_dir}/{file_}')

shutil.make_archive(output_dir,'zip',output_dir)

print(f'Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo_udith_github/ALAE/{output_dir}.zip ../../mnt/e/_PROJECTS/_ranga_sir/ALAE/tests')
