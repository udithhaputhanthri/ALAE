from defaults import get_cfg_defaults
from dataloader import *
import logging

cfg = get_cfg_defaults()
config_file = 'configs/citescapes.yaml'
cfg.merge_from_file(config_file)
cfg.freeze()

logger=  logging.getLogger("logger")
local_rank=0
world_size=1

dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=128, channels=cfg.MODEL.CHANNELS)
batch_size= 128
dataset.reset(7, 32)

batches = make_dataloader(cfg, logger, dataset, batch_size, local_rank)
x_orig = next(batches)

#import matplotlib.pyplot as plt
#plt.figure()
sample_img= x_orig[0].permute(1,2,0).cpu().detach().numpy()
#plt.imshow(sample_img.astype('uint8'))
print('batch_shape : ',x_orig.shape)
print('max, min in image : ',np.max(sample_img), np.min(sample_img))
print('loading succesful')
