import os.path as osp

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import sys  
import os, time
from ..bottlenecks import build_input_iba
from .base_attributor import BaseAttributor
from .builder import ATTRIBUTORS


@ATTRIBUTORS.register_module()
class VisionAttributor(BaseAttributor):

    def __init__(self,
                 layer: str,
                 classifier: dict,
                 feat_iba: dict,
                 input_iba: dict,
                 gan: dict,
                 use_softmax=True,
                 device='cuda:0'):
        super(VisionAttributor, self).__init__(
            layer=layer,
            classifier=classifier,
            feat_iba=feat_iba,
            input_iba=input_iba,
            gan=gan,
            use_softmax=use_softmax,
            device=device)

    def train_feat_iba(self, input_tensor, closure, attr_cfg, logger=None):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        feat_mask = self.feat_iba.analyze(
            input_tensor=input_tensor,
            model_loss_fn=closure,
            logger=logger,
            **attr_cfg)
        return feat_mask

    def train_input_iba(self,
                        input_tensor,
                        gen_input_mask,
                        closure,
                        attr_cfg,
                        logger=None):
        
        default_args = {
            'input_tensor': input_tensor,
            'input_mask': gen_input_mask
        }
        input_iba = build_input_iba(self.input_iba, default_args=default_args)
        
      
        
        _ = input_iba.analyze(input_tensor, closure, **attr_cfg, logger=logger)
        
       
        
        #MIL
        input_mask = torch.sigmoid(input_iba.alpha).detach().cpu().mean(
            [1]).numpy()
        
        
        
        return input_mask

    @staticmethod
    def get_closure(classifier, target, use_softmax, batch_size=None):
        if use_softmax:

            def closure(x):
            
                    
                
                print(batch_size)
                if x.dim() == 5:
                    losses = []
                    for i in range(x.shape[0]):
                        
                        losses.append(-torch.log_softmax(classifier(x[i]), 0)[target].unsqueeze(0))

                    loss = torch.cat(losses)
                
                
                else:
                    losses = []
                    for i in range(batch_size):
                        
                        losses.append(-torch.log_softmax(classifier(x), 0)[target].unsqueeze(0))

                    loss = torch.cat(losses)
                
                
                
                loss = loss.mean()
                return loss
        else:
            assert batch_size is not None
            # target is binary encoded and it is for a single sample
            assert isinstance(
                target,
                torch.Tensor) and target.max() <= 1 and target.dim() == 1
            raise NotImplementedError('Currently only support softmax')
        return closure

    def show_feat_mask(self, path, patient, upscale=True, show=False, out_file=None):
        if not upscale:
            mask = self.buffer['feat_iba_capacity']
        else:
            mask = self.buffer['feat_mask']
            
       
        
        path_feat = os.path.join(path, "feat")
        
        for i in range(mask.shape[0]):
             mask[i] = mask[i] / mask[i].max()
     
        self.show_mask(mask, path_feat, patient, show=show, out_file=out_file)

    def show_gen_input_mask(self, path, patient, show=False, out_file=None):
        mask = self.buffer['gen_input_mask']
        
        
        self.show_mask(mask, show=show, out_file=out_file)

    def show_input_mask(self, path, patient, show=False, out_file=None):
        mask = self.buffer['input_mask']
        path_iba = os.path.join(path, "input")
        
        
        self.show_mask(mask,path_iba, patient, show=show, out_file=out_file)

    @staticmethod
    def show_mask(mask, path, patient, show=False, out_file=None):
        
        np.save(os.path.join(path, patient + ".npy"), mask)
       
        mask_to_show = np.copy(mask)
        
        
        
        if mask.dtype in (float, np.float32, np.float16):
            mask = (mask * 255).astype(np.uint8)
        
        norm = colors.CenteredNorm(0)
        cm = plt.cm.get_cmap('bwr')
        mask_to_show = cm(norm(mask_to_show))
        

        if out_file is not None:
     
            plt.imsave(out_file + '.JPEG', mask_to_show)
        if not show:
            plt.close()
        else:
            plt.show()
       
            
    
