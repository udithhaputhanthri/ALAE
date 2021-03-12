import os
import matplotlib.pyplot as plt
import cv2


name='cityscapes'
type_='train'

dir_=name+'/'+name+'/'+type_

dir1= '/home/udith/data/datasets/citescapes-hq_raw'
dir2= '/home/udith/data/datasets/sementicmaps-hq_raw'


set_x=[]
set_y=[]

img_list=sorted(os.listdir(dir_))
def idx2imgname(idx, str_len=5, type_='jpg'):
    return (str_len-len(str(idx)))*'0'+str(idx)+'.'+type_

img_resize=256

for idx in range(len(img_list)):
    both=plt.imread(dir_+'/'+img_list[idx]).astype('uint8')
    x=cv2.resize(both[:,:both.shape[1]//2,:],(img_resize, img_resize))
    y=cv2.resize(both[:,both.shape[1]//2:,:],(img_resize, img_resize))

    plt.imsave(f'{dir1}/{idx2imgname(idx+1)}',x)
    plt.imsave(f'{dir2}/{idx2imgname(idx+1)}',y)
