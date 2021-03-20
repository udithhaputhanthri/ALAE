import torch

def get_loss(input_, target_, loss_type):
  batch_size = input_.shape[0]

  if loss_type=='l1':
    out= torch.nn.L1Loss()(input_, target_) #(INPUT, TARGET)
  if loss_type=='l2':
    out= torch.nn.MSELoss()(input_, target_) #(INPUT, TARGET)

  return out
