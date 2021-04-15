import torch
from pytorch_msssim import ssim, ms_ssim

def get_loss(input_, target_, loss_type):
  batch_size = input_.shape[0]
  if loss_type=='l1':
    out= torch.nn.L1Loss()(input_, target_) #(INPUT, TARGET)
  if loss_type=='l2':
    out= torch.nn.MSELoss()(input_, target_) #(INPUT, TARGET)
  if loss_type=='ssim':
    out =  1 - ssim(input_*255, target_*255, data_range=255, size_average=True) # return a scalar
  if loss_type=='ms_ssim':
    out = 1 - ms_ssim(input_*255, target_*255, data_range=255, size_average=True )



  return out
