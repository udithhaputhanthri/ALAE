
import cv2
import shutil
import numpy as np
import os
import matplotlib.pyplot as plt

def idx2imgname(idx, str_len=5, type_='jpg'):return (str_len-len(str(idx)))*'0'+str(idx)+'.'+type_

for i in range(150):
  j=np.random.randint(1, len(os.listdir('/home/udith/data/datasets/citescapes-hq_raw')))

  img_name=f'/home/udith/data/datasets/citescapes-hq_raw/{idx2imgname(j)}'

  plt.imsave(f'dataset_samples/citescapes-hq/{idx2imgname(i+1)}',cv2.resize(plt.imread(img_name),(256, 256)))
  plt.imsave(f'dataset_samples/citescapes/{idx2imgname(i+1)}',cv2.resize(plt.imread(img_name),(128, 128)))

print(len(os.listdir('dataset_samples/citescapes/')), len(os.listdir('dataset_samples/citescapes-hq/')))
print(plt.imread('dataset_samples/citescapes/00001.jpg').shape, plt.imread('dataset_samples/citescapes-hq/00001.jpg').shape)
