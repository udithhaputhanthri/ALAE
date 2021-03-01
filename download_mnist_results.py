import shutil
import os
try:os.mkdir('mnist_results_images')
except:pass

for file_ in os.listdir('mnist_results'):
    if '.jpg' in file_:
        shutil.copy(f'mnist_results/{file_}',f'mnist_results_images/{file_}')

shutil.make_archive('mnist_results_download','zip','mnist_results_images')

print('Download results ::: scp -i ~/.ssh/ranga_sir_server/id_rsa udith@192.248.10.120:alae_official_repo/ALAE/mnist_results_download.zip .')
