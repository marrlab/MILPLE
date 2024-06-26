import warnings

import mmcv
import torch
import torch.nn as nn
from tqdm import tqdm

from ..estimators import VisionWelfordEstimator
from ..utils import _SpatialGaussianKernel
from .base_feat_iba import BaseFeatureIBA


class VisionFeatureIBA(BaseFeatureIBA):
    """
        iba finds relevant features of your model by applying noise to
        intermediate features.
    """

    def __init__(self, **kwargs):
        super(VisionFeatureIBA, self).__init__(**kwargs)

    @torch.no_grad()
    def reset_alpha(self):
        self.alpha.fill_(self.initial_alpha)

    def init_alpha_and_kernel(self):
        # TODO to check if it is neccessary to keep it in base class
        if self.estimator.n_samples() <= 0:
            raise RuntimeWarning(
                "You need to estimate the feature distribution"
                " before using the bottleneck.")
        shape = self.estimator.shape
        device = self.estimator.device
        self.alpha = nn.Parameter(
            torch.full(shape, self.initial_alpha, device=device),
            requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(
                2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma,
                                                 shape[0]).to(device)
        else:
            self.smooth = None

    def reset_estimator(self):
        self.estimator = VisionWelfordEstimator()

    def estimate(self,
                 model,
                 dataloader,
                 n_samples=10000,
                 verbose=False,
                 reset=True):
        if verbose:
            bar = tqdm(dataloader, total=n_samples)
        else:
            bar = None

        if reset:
            self.reset_estimator()
        count = 0   
        for batch in dataloader:
            
            if isinstance(batch, tuple) or isinstance(batch, list):
                imgs = batch[0]
            else:
                imgs = batch['input']
            
            count+=1
            
            if self.estimator.n_samples() > n_samples*1000:
                
            
                
                break
            with torch.no_grad(), self.interrupt_execution(
            ), self.enable_estimation():
                
                
                imgs = imgs.squeeze()
                
                
                imgs = imgs.float()
                model(imgs)
                
                
                if bar:
                    bar.update(len(imgs))
        if bar:
            bar.close()

        # Cache results
        self.input_mean = self.estimator.mean()
        self.input_std = self.estimator.std()
        self.active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        # After estimaton, feature map dimensions are known and
        # we can initialize alpha and the smoothing kernel
        if self.alpha is None:
            self.init_alpha_and_kernel()
        

    def do_restrict_info(self, x, alpha):
        
        #MIL
        batch_size=10
        
        if alpha is None:
            raise RuntimeWarning("Alpha not initialized. Run "
                                 "init_alpha_and_kernel() "
                                 "before using the bottleneck.")

        if self.input_mean is None:
            self.input_mean = self.estimator.mean()

        if self.input_std is None:
            self.input_std = self.estimator.std()

        if self.active_neurons is None:
            self.active_neurons = self.estimator.active_neurons()

        # Smoothen and expand alpha on batch dimension
        lamb = torch.sigmoid(alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb
        
          
         
        
           
   
        
        buffer_capacity = self.kl_div(
            x, lamb, self.input_mean, self.input_std) * self.active_neurons
     
        
        
        if self.buffer_capacity is not None:
            buffer_capacity = buffer_capacity.unsqueeze(0)
            self.buffer_capacity = torch.cat((self.buffer_capacity, buffer_capacity))
        else:
            self.buffer_capacity = buffer_capacity.unsqueeze(0)
        
       
        eps = x.data.new(x.size()).normal_()
        ε = self.input_std.to(x.device) * eps + self.input_mean.to(x.device)
      
        λ = lamb.to(x.device)
        if self.reverse_lambda:
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε
        z *= self.active_neurons.to(x.device)

        # Sample new output values from p(z|x)

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)
            
            
            
        torch.cuda.empty_cache()

        return z

    def analyze(  # noqa
            self,
            input_tensor,
            model_loss_fn,
            mode='saliency',
            beta=10.0,
            opt_steps=10,
            lr=1.0,
            batch_size=10,
            min_std=0.01,
            logger=None,
            log_every_steps=-1):
        
        
        
        
        if logger is None:
            logger = mmcv.get_logger('input_iba')
        
        batch = input_tensor
        

        # Reset from previous run or modifications
        self.reset_alpha()
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.estimator.n_samples() < 1000:
            warnings.warn(f"Selected estimator was only fitted "
                          f"on {self.estimator.n_samples()} samples. Might "
                          f"not be enough! We recommend 10.000 samples.")
        std = self.estimator.std()
        self.active_neurons = self.estimator.active_neurons(
            self._active_neurons_threshold).float()
        self.input_std = torch.max(std, min_std * torch.ones_like(std))

        self.reset_loss_buffers()

        with self.restrict_flow():
            for i in range(opt_steps):
                optimizer.zero_grad()
                cls_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling the sum with 1/K
               
                info_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -cls_loss + beta * info_loss
                else:
                    loss = cls_loss + beta * info_loss
                loss.backward()
                optimizer.step()

                self.loss_buffer.append(loss.item())
                self.cls_loss_buffer.append(cls_loss.item())
                self.info_loss_buffer.append(info_loss.item())
                if log_every_steps > 0 and (i + 1) % log_every_steps == 0:
                    log_str = f'Feature IBA: step [{i + 1}/ {opt_steps}], '
                    log_str += f'loss: {self.loss_buffer[-1]:.5f}, '
                    log_str += f'cls loss: {self.cls_loss_buffer[-1]:.5f}, '
                    log_str += f'info loss: {self.info_loss_buffer[-1]:.5f}'
                    logger.info(log_str)
                
                #MIL
                
                if i != (opt_steps - 1):
                    self.buffer_capacity = None
                
                self.count=1
                torch.cuda.empty_cache()
                
        
        return self._get_saliency(mode=mode, shape=input_tensor.shape[2:])
        
