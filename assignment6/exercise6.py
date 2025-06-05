import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim

class SRDataset(Dataset):
  def __init__(self, folder_path, transform=None, low_res_transform=None):
    """
    Args:
      folder_path (string): path to the image folder.
      transform (callable): transform to be applied on image samples.
      low_res_transform (callable): transform to create low-resolution images.
    """
    self.folder_path = folder_path
    self.transform = transform
    self.low_res_transform = low_res_transform
    self.image_filenames = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('jpg', 'jpeg', 'png'))]

  def __len__(self):
    return len(self.image_filenames)

  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()

    image_name = self.image_filenames[index]

    # load image as a tensor and convert it to float between 0 and 1
    image = io.read_image(image_name)
    image = image.float() / 255.0

    if self.transform:
      image = self.transform(image)

    low_res_image = None
    if self.low_res_transform:
      low_res_image = self.low_res_transform(image)

    return image, low_res_image

# the transformation pipeline for the high-resolution images
transform = transforms.Compose([
    transforms.RandomCrop(64),           # randomly crop a 64x64 patch from the image
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)  # use ColorJitter to change the brightness, contrast, saturation, and hue
])

# the transformation pipeline for the low-resolution images
low_res_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR)  # downscale to 32x32
])


class BasicSRModel(nn.Module):
  def __init__(self, num_blocks = 10):
    super().__init__()

    # first upsample bilinearly the low resolutino image according to the scale factor
    self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    # first convolution: input 3 - output 64
    self.first_convolution = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)

    # i-th block’s convolution: input 64 - output 64
    self.block_convolution = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1), nn.LeakyReLU()) for _ in range(num_blocks)])

    # last convolution: input 64 - output 3
    self.last_convolution = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1)

  def forward(self, x):
    # bilinear upsampling
    x = self.upsample(x)

    # first convolution
    x = self.first_convolution(x)
    
    # block convolutions
    x = self.block_convolution(x)
    
    # last convolution
    x = self.last_convolution(x)
    return x 
  
# residual model, the only difference is the last line of the forward method  
class ResidualModel(nn.Module):
  def __init__(self, num_blocks = 10):
    super().__init__()

    # first upsample bilinearly the low resolutino image according to the scale factor
    self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    # first convolution: input 3 - output 64
    self.first_convolution = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)

    # i-th block’s convolution: input 64 - output 64
    self.block_convolution = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1), nn.LeakyReLU()) for _ in range(num_blocks)])

    # last convolution: input 64 - output 3
    self.last_convolution = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1)

  def forward(self, x):
    # bilinear upsampling
    upsampled = self.upsample(x)

    # first convolution
    first_convolved = self.first_convolution(upsampled)
    
    # block convolutions
    block_convolved = self.block_convolution(first_convolved)
    
    # last convolution
    last_convolved = self.last_convolution(block_convolved)

    # residual connection
    output = upsampled + last_convolved
    return output   

  
# calculate the psnr loss for a batch of images 
# return average among images in the batch
def psnr_metric(images1, images2):
  mse_loss = nn.MSELoss()(images1, images2)
  psnr = -10 * torch.log10(mse_loss)
  return psnr.item()

# calculate the ssim loss for a batch of images
# return average among images in the batch
def ssim_metric(images1, images2):
  ssim_value = ssim(images1, images2, data_range = 1.0, size_average = True)
  return ssim_value.item()


def evaluate_model(model, dataloader, device):
  model.eval()
  total_psnr_bilinear = 0
  total_psnr_bicubic = 0
  total_psnr_model = 0
  total_ssim_bilinear = 0
  total_ssim_bicubic = 0
  total_ssim_model = 0
  num_batches = 0

  with torch.no_grad():
    for high_res, low_res in dataloader:
      high_res = high_res.to(device)
      low_res = low_res.to(device)

      # traditional bilinear upscaling
      upscaled_bilinear = nn.functional.interpolate(low_res, scale_factor=2, mode='bilinear')
      psnr_bilinear = psnr_metric(high_res, upscaled_bilinear)
      ssim_bilinear = ssim_metric(high_res, upscaled_bilinear)
      total_psnr_bilinear += psnr_bilinear
      total_ssim_bilinear += ssim_bilinear

      # traditional bicubic upscaling
      upscaled_bicubic = nn.functional.interpolate(low_res, scale_factor=2, mode='bicubic')
      psnr_bicubic = psnr_metric(high_res, upscaled_bicubic)
      ssim_bicubic = ssim_metric(high_res, upscaled_bicubic)
      total_psnr_bicubic += psnr_bicubic
      total_ssim_bicubic += ssim_bicubic

      # model-based upscaling
      upscaled_model = model(low_res)
      psnr_model = psnr_metric(high_res, upscaled_model)
      ssim_model = ssim_metric(high_res, upscaled_model)
      total_psnr_model += psnr_model
      total_ssim_model += ssim_model

      num_batches += 1

  avg_psnr_bilinear = total_psnr_bilinear / num_batches
  avg_psnr_bicubic = total_psnr_bicubic / num_batches
  avg_psnr_model = total_psnr_model / num_batches
  avg_ssim_bilinear = total_ssim_bilinear / num_batches
  avg_ssim_bicubic = total_ssim_bicubic / num_batches
  avg_ssim_model = total_ssim_model / num_batches

  print(f'bilinear psnr loss: {avg_psnr_bilinear:.4f}')
  print(f'bicubic psnr loss: {avg_psnr_bicubic:.4f}')
  print(f'model psnr loss: {avg_psnr_model:.4f}')
  print(f'bilinear ssim loss: {avg_ssim_bilinear:.4f}')
  print(f'bicubic ssim loss: {avg_ssim_bicubic:.4f}')
  print(f'model ssim loss: {avg_ssim_model:.4f}')
    
  return avg_psnr_bilinear, avg_psnr_bicubic, avg_psnr_model, avg_ssim_bilinear, avg_ssim_bicubic, avg_ssim_model



if __name__ == '__main__':

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))

  # create the train_dataset
  train_dataset = SRDataset(folder_path=r'D:\2024FS\MathFound\EX6\MF2024-Exercise6\data\train', transform=transform, low_res_transform=low_res_transform)

  # Task1
  # create the train_dataloader
  train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

  #print(f"* Dataset contains {len(train_dataset)} image(s).")

  #for _ , batch in enumerate(train_dataloader, 0):
    #hr_image, lr_image = batch
    #io.write_png(lr_image[0,...].mul(255).byte(), "lr_image.png")
    #io.write_png(hr_image[0,...].mul(255).byte(), "hr_image.png")
    #ybreak # we deliberately break after one batch as this is just a test


  # Task2
  # change BasicSRModel to ResidualModel to train residual model
  model = BasicSRModel(10).to(device)
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  print(num_params)

  # Task4
  # transformation for eval low_res images, change BILINEAR to BICUBIC or NEAREST for evaluating different downscaling methods
  eval_low_res_transform = transforms.Compose([transforms.Resize((240, 240), interpolation=transforms.InterpolationMode.BILINEAR)])

  # create the eval_dataset
  eval_dataset = SRDataset(folder_path=r'D:\2024FS\MathFound\EX6\MF2024-Exercise6\data\eval', transform=None, low_res_transform=eval_low_res_transform)

  # create the train_dataloader
  eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)   

  # Task3
  # change learing_rate to test different learning rates
  learning_rate = 1e-4
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
  loss_function = torch.nn.L1Loss().to(device)

  writer = SummaryWriter('runs/')

 # Now number_of_epochs = 0 is set for not training
  number_of_epochs = 0
  save_model_interval = 2

  for epoch in range(number_of_epochs):
    model.train()
    running_loss = 0.0
    for i , (high_res, low_res) in enumerate(train_dataloader):
      high_res = high_res.to(device)
      low_res = low_res.to(device)

      # reset the gradient
      optimizer.zero_grad()

      # forward pass through the model
      high_res_prediction = model(low_res)

      # compute the loss
      loss = loss_function(high_res_prediction, high_res)

      # backpropagation
      loss.backward()

      # update the model parameters
      optimizer.step()

      # accumulate loss
      running_loss += loss.item()

      if i % 10 == 9:
        print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
        writer.add_scalar('training loss', running_loss / 10, epoch * len(train_dataloader) + i)
        running_loss = 0.0
    
    # Task4
    # evalution
    model.eval()
    val_loss_l1 = 0.0
    val_loss_psnr = 0.0
    val_loss_ssim = 0.0
    with torch.no_grad():
      for high_res, low_res in eval_dataloader:
        high_res = high_res.to(device)
        low_res = low_res.to(device)
        outputs = model(low_res)
        loss_l1 = loss_function(outputs, high_res)
        loss_psnr = psnr_metric(outputs, high_res)
        loss_ssim = ssim_metric(outputs, high_res)
        val_loss_l1 += loss_l1.item()
        val_loss_psnr += loss_psnr
        val_loss_ssim += loss_ssim
    
    avg_val_loss_l1 = val_loss_l1 / len(eval_dataloader)
    avg_val_loss_psnr = val_loss_psnr / len(eval_dataloader)
    avg_val_loss_ssim = val_loss_ssim / len(eval_dataloader)
    print(f'[Epoch {epoch + 1}] L1 validation loss: {avg_val_loss_l1:.3f}')
    writer.add_scalar('L1 validation loss', avg_val_loss_l1, epoch)
    print(f'[Epoch {epoch + 1}] PSNR validation loss: {avg_val_loss_psnr:.3f}')
    writer.add_scalar('PSNR validation loss', avg_val_loss_psnr, epoch)
    print(f'[Epoch {epoch + 1}] SSIM validation loss: {avg_val_loss_ssim:.3f}')
    writer.add_scalar('SSIM validation loss', avg_val_loss_ssim, epoch)

    # Task3
    if (epoch + 1) % save_model_interval == 0:
      torch.save(model.state_dict(), f'checkpoint/checkpoint_epoch_{epoch + 1}.pth')   

  print('Finished Training')
  writer.close() 

  # Task4
  eval_model = BasicSRModel(10).to(device)
  eval_model.load_state_dict(torch.load('checkpoint/checkpoint_epoch_290.pth'))
  eval_model.eval()
  evaluate_model(eval_model, eval_dataloader, device)

  # process and save the images upsampled using traditional bilinear and bicubic interpolation methods
  # used for visual comparision with CNN
  # uncomment if you would like to try
  #for batch_idx , batch in enumerate(eval_dataloader):
  #  _, lr_images = batch
  #  bilinear_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
  #  bicubic_upsample = nn.Upsample(scale_factor=2, mode='bicubic')
    
  #  for i, lr_image in enumerate(lr_images):
  #    # apply upsampling
  #    upsampled_bilinear = bilinear_upsample(lr_image.unsqueeze(0)).squeeze(0)
  #    upsampled_bicubic = bicubic_upsample(lr_image.unsqueeze(0)).squeeze(0)
        
  #    bilinear_image_byte = (torch.clamp(upsampled_bilinear * 255, 0, 255)).byte()
  #    bicubic_image_byte = (torch.clamp(upsampled_bicubic * 255, 0, 255)).byte()
            
  #    bilinear_filename = os.path.join(r'D:\2024FS\MathFound\EX6\MF2024-Exercise6\bilinear_task4', f'lr_image_bilinear_{batch_idx}_{i}.png')
  #    bicubic_filename = os.path.join(r'D:\2024FS\MathFound\EX6\MF2024-Exercise6\bicubic_task4', f'lr_image_bicubic_{batch_idx}_{i}.png')
        
  #    # Save images
  #    io.write_png(bilinear_image_byte, bilinear_filename)
  #    io.write_png(bicubic_image_byte, bicubic_filename)

  #print(f'Saved bilinear and bicubic interpolated files')


  # process and save the images by CNN
  for batch_idx , batch in enumerate(eval_dataloader):
    _, lr_images = batch
    upsampled_model = eval_model(lr_images.to(device)).to('cpu')

    for i, image in enumerate(upsampled_model):
      model_image_byte = (torch.clamp(image * 255.0, 0, 255)).byte()
      model_filename = os.path.join(r'D:\2024FS\MathFound\EX6\MF2024-Exercise6\model_task4', f'lr_image_model_{batch_idx}_{i}.png')
      io.write_png(model_image_byte, model_filename)

  
