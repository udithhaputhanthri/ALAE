
import cv2
import shutil
import numpy as np
import os
import matplotlib.pyplot as plt

def idx2imgname(idx, str_len=5, type_='jpg'):return (str_len-len(str(idx)))*'0'+str(idx)+'.'+type_

for i in range(150):
  j=np.random.randint(1, len(os.listdir('/home/udith/data/datasets/birds/train/images')))

  img_name=f'/home/udith/data/datasets/birds/train/images/{idx2imgname(j)}'

  plt.imsave(f'dataset_samples/stackGANbirds/{idx2imgname(i+1)}',plt.imread(img_name))


print(len(os.listdir('dataset_samples/stackGANbirds/')))
print(plt.imread('dataset_samples/stackGANbirds/00001.jpg').shape)
