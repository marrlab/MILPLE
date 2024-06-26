import random

import torch
from mmcv import get_logger
from torch.optim import RMSprop
from torch.utils.data import DataLoader, Dataset

from .base_gan import BaseWassersteinGAN, _get_gan_log_string
from .builder import GANS, build_discriminator, build_generator
import torch.nn as nn


@GANS.register_module()
class VisionWGAN(BaseWassersteinGAN):

    def __init__(self,
                 generator: dict,
                 discriminator: dict,
                 input_tensor,
                 context,
                 feat_mask=None,
                 device='cuda:0'):
        super(VisionWGAN, self).__init__(
            input_tensor=input_tensor,
            context=context,
            generator=generator,
            discriminator=discriminator,
            feat_mask=feat_mask,
            device=device)

    def build_generator(self, input_tensor, context, cfg):
        default_args = {
            'input_tensor': self.input_tensor,
            'context': context,
            'device': self.device,
            'capacity': self.feat_mask
        }
        return build_generator(cfg, default_args=default_args)

    def build_discriminator(self, cfg):
        default_args = {
            #Single Instance
            'channels': self.feat_map.shape[0],
            'size': self.feat_map.shape[1:]
            
            
        }
        return build_discriminator(cfg, default_args=default_args)

    def build_data(self, dataset_size, sub_dataset_size, batch_size):
        # create dataset from feature mask and feature map
        dataset = VisionSyntheticDataset(self.feat_map, self.feat_mask,
                                         dataset_size, sub_dataset_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True)
        return dataloader

    def train(self,
              dataset_size=200,
              sub_dataset_size=20,
              lr=5e-5,
              batch_size=32,
              weight_clip=0.1,
              epochs=200,
              critic_iter=5,
              verbose=False,
              logger=None):
        # TODO add learning rate scheduler
        # Initialize generator and discriminator
        
        self.device = "cuda"
        
        
         
       
        
        #print("devide for gan ", self.device)
        if logger is None:
            logger = get_logger('input_iba')
        data_loader = self.build_data(dataset_size, sub_dataset_size,
                                      batch_size)

        # Optimizers
        optimizer_G = RMSprop([{
            "params": self.generator.mean,
            "lr": 0.1
        }, {
            "params": self.generator.eps,
            "lr": 0.05
        }, {
            "params": self.generator.input_mask_param,
            "lr": 0.003
        }])
        optimizer_D = RMSprop(self.discriminator.parameters(), lr=lr)
        
       

        # training
        num_iters = 0
        for epoch in range(epochs):
            #print("epoch running ", epoch)
            for i, data in enumerate(data_loader):
                torch.cuda.empty_cache()

                # train discriminator
                # Deprecated: if dataloader loads TensorDataset, then data is a tuple of one element
                real_input = data
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.zeros_like(self.input_tensor)
                
                #Single Instance
                z = z.unsqueeze(0).expand(real_input.shape[0], -1, -1,
                                          -1).clone().normal_().to(self.device) CHANGE FOR CLUSTER
             
                # Generate a batch of images
                fake_input = self.generator(z).detach()
               
                # Adversarial loss
                
                
               
                
                #Single Instance
                loss_D = -torch.mean(
                    self.discriminator(real_input)) + torch.mean(
                        self.discriminator(fake_input))
                       
                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

                # Train the generator every n_critic iterations
                if i % critic_iter == 0:
                    # train generator
                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_input = self.generator(z)
                    
                    
                    # Adversarial loss
                   
                    
                    #Single Instance
                    loss_G = -torch.mean(self.discriminator(gen_input))
                    #

                    loss_G.backward()
                    optimizer_G.step()
                    if verbose:
                        log_str = _get_gan_log_string(epoch + 1, epochs,
                                                      num_iters + 1,
                                                      len(data_loader),
                                                      loss_D.item(),
                                                      loss_G.item())
                        logger.info(log_str)
                num_iters += 1
                

        del data_loader
        self.generator.clear()


class VisionSyntheticDataset(Dataset):
    """create dataset from feature mask and feature map"""

    def __init__(self, feature_map, feature_mask, dataset_size,
                 sub_dataset_size):
        """
        Args:
            feature_map (torch.Tensor): feature map of input
            feature_mask (torch.Tensor): learned feature mask from IBA
            dataset_size (int): size of synthetic dataset
            sub_dataset_size (int): size of subdataset, each subdataset takes different mean and var
            seed (int): a seed for the random number generator
        """
        self.feature_map = feature_map
        self.feature_mask = feature_mask
        self.dataset_size = dataset_size
        self.sub_dataset_size = sub_dataset_size

        # initialize a fixed set of means and stds
        num_sub_dataset = int(dataset_size / sub_dataset_size)
        self.std = [random.uniform(0, 5) for i in range(num_sub_dataset)]
        self.mean = [random.uniform(-2, 2) for i in range(num_sub_dataset)]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # get data from synthetic dataset based on masking scheme
        idx_sub_dataset = int(idx / self.sub_dataset_size)
        noise = torch.zeros_like(self.feature_mask).normal_()
        noise = self.std[idx_sub_dataset] * noise + self.mean[idx_sub_dataset]
        masked_feature = self.feature_mask.to(noise.device) * self.feature_map.to(noise.device) + (
            1 - self.feature_mask.to(noise.device)) * noise
        return masked_feature.detach()
