'''Denoising diffusion model.'''

import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .unet import UNet
from .schedules import make_beta_schedule


class DDPM(pl.LightningModule):
    '''
    Plain vanilla DDPM module.

    Summary
    -------
    This module establishes a plain vanilla DDPM variant.
    It is basically a container and wrapper for an
    epsilon model and for the scheduling parameters.
    The class provides methods implementing the forward
    and reverse diffusion processes, respectively.
    Also, the stochastic loss can be computed for training.

    Parameters
    ----------
    eps_model : PyTorch module
        Trainable noise-predicting model.
    betas : array-like
        Beta parameter schedule.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Optimizer learning rate.

    '''

    def __init__(self,
                 eps_model,
                 betas,
                 criterion='mse',
                 lr=1e-04,
                 reg=0.01):
        super().__init__()

        # set trainable epsilon model
        self.eps_model = eps_model

        # set loss function criterion
        if criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError('Criterion could not be determined')

        # set initial learning rate
        self.lr = abs(lr)

        # set arrays for iso_difference, eps_pred and eps
        self.iso_difference_list = []             
        self.eps_pred_list = []
        self.eps_list = []   


        self.reg = reg
      
                     

        # to save losses
        self.train_losses = []
        self.val_losses = []

        self.norms = []

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
          
        

        # set scheduling parameters
        betas = torch.as_tensor(betas).view(-1) # note that betas[0] corresponds to t = 1.0

        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError('Invalid beta values encountered')

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(betas_tilde, pad=(1, 0), value=0.0) # ensure betas_tilde[0] = 0.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_tilde', betas_tilde)

    @property
    def num_steps(self):
        '''Get the total number of time steps.'''
        return len(self.betas)

    def forward(self, x, t):
        '''Run the noise-predicting model.'''
        return self.eps_model(x, t)

    def diffuse_step(self, x, tidx):
        '''Simulate single forward process step.'''
        beta = self.betas[tidx]
        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps
        return x_noisy

    def diffuse_all_steps_till_time(self, x0, time):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(time + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(time):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse_all_steps(self, x0):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape, device=x0.device)
        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse(self, x0, tids, return_eps=False):
        '''Simulate multiple forward steps at once.'''
        alpha_bar = self.alphas_bar[tids]
        eps = torch.randn_like(x0)

        missing_shape = [1] * (eps.ndim - alpha_bar.ndim)
        alpha_bar = alpha_bar.view(*alpha_bar.shape, *missing_shape)

        x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

        if return_eps:
            return x_noisy, eps
        else:
            return x_noisy

    def denoise_step(self, x, tids, random_sample=False):
        '''Perform single reverse process step.'''
        # set up time variables
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1) # ensure (batch_size>=1, 1)-shaped tensor
        ts = tids.to(x.dtype) + 1 # note that tidx = 0 corresponds to t = 1.0

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x, ts)

        # compute mean
        p = 1 / self.alphas[tids].sqrt()
        q = self.betas[tids] / (1 - self.alphas_bar[tids]).sqrt()

        missing_shape = [1] * (eps_pred.ndim - ts.ndim)
        p = p.view(*p.shape, *missing_shape)
        q = q.view(*q.shape, *missing_shape)

        x_denoised_mean = p * (x - q * eps_pred)

        # retrieve variance
        x_denoised_var = self.betas_tilde[tids]
        # x_denoised_var = self.betas[tids]

        # generate random sample
        if random_sample:
            eps = torch.randn_like(x_denoised_mean)
            x_denoised = x_denoised_mean + x_denoised_var.sqrt() * eps

        if random_sample:
            return x_denoised
        else:
            return x_denoised_mean, x_denoised_var

    @torch.no_grad()
    def denoise_all_steps(self, xT):
        '''Perform and return all reverse process steps.'''
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)

        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):
            # generate random sample
            if tidx > 0:
                x_denoised[idx + 1] = self.denoise_step(x_denoised[idx], tidx, random_sample=True)
            # take the mean in the last step
            else:
                x_denoised[idx + 1], _ = self.denoise_step(x_denoised[idx], tidx, random_sample=False)

        return x_denoised

    @torch.no_grad()
    def generate(self, sample_shape, num_samples=1):
        '''Generate random samples through the reverse process.'''
        x_denoised = torch.randn(num_samples, *sample_shape, device=self.device) # Lightning modules have a device attribute
        isotropy = []
        for tidx in reversed(range(self.num_steps)):
            # generate random sample
            if tidx > 0:
                x_denoised = self.denoise_step(x_denoised, tidx, random_sample=True)
                iso = self.isotropy(x_denoised)
                isotropy.append(iso)
            # take the mean in the last step
            else:
                x_denoised, _ = self.denoise_step(x_denoised, tidx, random_sample=False)
                iso = self.isotropy(x_denoised)
                isotropy.append(iso)

        return x_denoised, isotropy
    

    def isotropy(self, data):
        data = data.detach().cpu().numpy()
        iso = np.vdot(data, data) / len(data)
        return iso
    
    def loss(self, x):
        '''Compute stochastic loss.'''
        # # draw random time steps
        tids = torch.randint(0, self.num_steps, size=(x.shape[0], 1), device=x.device)
        
        ts = tids.to(x.dtype) + 1 # note that tidx = 0 corresponds to t = 1.0
        
        # perform forward process steps
        x_noisy, eps = self.diffuse(x, tids, return_eps=True)
        # self.eps_list.append(eps)

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x_noisy, ts)
        # self.eps_pred_list.append(eps_pred)

        # compute squared norm loss
        # dim_ = torch.tensor(2.0, requires_grad=True)
        # squared_norm_preds = torch.mean(torch.sum(eps_pred**2, dim=2)) / dim_
        
        # compute the covariance matrix
        covariance_matrix = torch.matmul(eps_pred.T, eps_pred)
        
        diag_elements_mean = covariance_matrix.diagonal().mean()
        non_diag_elements_mean = torch.mean(torch.tril(covariance_matrix, diagonal=-1))
        
        log_loss = torch.log(1 - non_diag_elements_mean / diag_elements_mean)

        log_loss_target = torch.tensor(0.0, device=eps_pred.device)
        
        norm_loss = self.criterion(log_loss.to(eps_pred.device), log_loss_target.to(eps_pred.device))
        simple_diff_loss = self.criterion(eps_pred, eps)
        
        loss = simple_diff_loss + self.reg*norm_loss 

        return loss, simple_diff_loss, norm_loss

    def train_step(self, x_batch):
        self.optimizer.zero_grad()
        loss, simple_diff_loss, norm_loss = self.loss(x_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), simple_diff_loss.item(), norm_loss.item()

    def get_eps_pred_list(self):
        return self.eps_pred_list

    def get_eps_list(self):
        return self.eps_list

    def get_iso_difference_list(self):
        return self.iso_difference_list
    
    def validate(self, val_loader):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val_batch in val_loader:
                val_loss, _, _ = self.loss(torch.stack(x_val_batch)).item()
                val_loss += val_loss
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss


class DDPM2d(DDPM):
    '''
    DDPM for problems with two spatial dimensions.

    Summary
    -------
    This subclass facilitates the construction of a 2D DDPM.
    It consists of U-net-based noise model and a beta schedule.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : list or tuple of ints
        Hidden layer channel numbers.
    kernel_size : int
        Convolutional kernel size.
    padding : int
        Padding parameter.
    norm : str
        Normalization type.
    activation : str
        Nonlinearity type.
    embed_dim : int
        Embedding dimension.
    num_resblocks : int
        Number of residual blocks.
    upsample_mode : str
        Convolutional upsampling mode.
    beta_mode : str
        Beta schedule mode.
    beta_range: (float, float)
        Beta range for linear and quadratic schedule.
    cosine_s : float
        Offset for cosine-based schedule.
    sigmoid_range : (float, float)
        Sigmoid range for sigmoid-based schedule.
    num_steps : int
        Number of time steps.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Optimizer learning rate.

    '''

    def __init__(self,
                 in_channels=1,
                 mid_channels=[16, 32, 64],
                 kernel_size=3,
                 padding=1,
                 norm='batch',
                 activation='leaky_relu',
                 embed_dim=128,
                 num_resblocks=3,
                 upsample_mode='conv_transpose',
                 beta_mode='cosine',
                 beta_range=[1e-04, 0.02],
                 cosine_s=0.008,
                 sigmoid_range=[-5, 5],
                 num_steps=1000,
                 criterion='mse',
                 lr=1e-04):

        # construct U-net model
        eps_model = UNet.from_params(
            in_channels=in_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            activation=activation,
            embed_dim=embed_dim,
            num_resblocks=num_resblocks,
            upsample_mode=upsample_mode
        )

        # create noise schedule
        beta_opts = {}
        if beta_mode in ('linear', 'quadratic'):
            beta_opts['beta_range'] = beta_range
        elif beta_mode == 'cosine':
            beta_opts['cosine_s'] = cosine_s
        elif beta_mode == 'sigmoid':
            beta_opts['sigmoid_range'] = sigmoid_range

        betas = make_beta_schedule(num_steps, mode=beta_mode, **beta_opts)

        # initialize DDPM class
        self.save_hyperparameters() # write hyperparams to checkpoints

        super().__init__(
            eps_model=eps_model,
            betas=betas,
            criterion=criterion,
            lr=lr
        )
