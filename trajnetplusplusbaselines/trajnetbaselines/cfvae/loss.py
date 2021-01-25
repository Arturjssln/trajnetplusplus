"""Loss used in the VAE model"""

import math
import torch

class KLDLoss(torch.nn.Module):
    """Kullback-Leibler divergence Loss

    """
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, inputs, targets=None):
        """
        Forward path of Kullback-Leibler divergence Loss

        Parameters:
        -----------
        inputs : Tensor [batch_size, 2*latent_dim]
            Tensor containing multivariate distribution mean and logarithmic variance
        targets : Tensor [batch_size, 2*latent_dim]
            Tensor containing target multivariate distribution mean and logarithmic variance
            Default: standard normal distribution (zero mean and unit variance)
        
        Output:
        -----------
        loss : Tensor [1]
            Tensor containing Kullback-Leibler divergence loss
        """

        if targets is None:
            # Default KLD Loss (with standard normal distribution, simplified equation)
            z_mu, z_log_var = torch.split(inputs, split_size_or_sections=inputs.size(1)//2, dim=1)
            latent_loss = -0.5 * torch.sum(1.0 + z_log_var - torch.square(z_mu) - torch.exp(z_log_var), dim=1)
        else:
            # KLD Loss between the distributions inputs and targets
            z_mu, z_log_var = torch.split(inputs, split_size_or_sections=inputs.size(1)//2, dim=1)
            z_mu_t, z_log_var_t = torch.split(targets, split_size_or_sections=targets.size(1)//2, dim=1)
            z_var = torch.exp(z_log_var)
            z_var_t = torch.exp(z_log_var_t)
            latent_dim = z_mu.size(1)
            latent_loss = 0.5 * (((1/z_var_t)*z_var).sum(dim=1) + ((z_mu_t-z_mu)**2 * z_var_t).sum(dim=1)\
                - latent_dim + torch.log(torch.prod(z_var_t, dim=1)/torch.prod(z_var, dim=1)))
        return torch.mean(latent_loss)
    

class PredictionLoss(torch.nn.Module):
    """2D Gaussian with a flat background.

    p(x) = 0.2 * N(x|mu, 3.0)  +  0.8 * N(x|mu, sigma)
    """
    def __init__(self, keep_batch_dim=False, background_rate=0.2):
        super(PredictionLoss, self).__init__()
        self.keep_batch_dim = keep_batch_dim
        self.background_rate = background_rate

    @staticmethod
    def gaussian_2d(mu1mu2s1s2rho, x1x2):
        """This supports backward().

        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """

        x1, x2 = x1x2[:, 0], x1x2[:, 1]
        mu1, mu2, s1, s2, rho = (
            mu1mu2s1s2rho[:, 0],
            mu1mu2s1s2rho[:, 1],
            mu1mu2s1s2rho[:, 2],
            mu1mu2s1s2rho[:, 3],
            mu1mu2s1s2rho[:, 4],
        )

        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return numerator / denominator

    def forward(self, inputs, targets, batch_split):
        
        pred_length, batch_size = targets.size(0), batch_split[:-1].size(0)
        ## Extract primary pedestrians
        # [pred_length, num_tracks, 2] --> [pred_length, batch_size, 2]
        targets = targets.transpose(0, 1)
        targets = targets[batch_split[:-1]]
        targets = targets.transpose(0, 1)

        # [pred_length, num_tracks, 5] --> [pred_length, batch_size, 5]
        inputs = inputs.transpose(0, 1)
        inputs = inputs[batch_split[:-1]]
        inputs = inputs.transpose(0, 1)

        ## Loss calculation
        inputs = inputs.reshape(-1, 5)
        targets = targets.reshape(-1, 2)
        inputs_bg = inputs.clone()
        inputs_bg[:, 2] = 3.0  # sigma_x
        inputs_bg[:, 3] = 3.0  # sigma_y
        inputs_bg[:, 4] = 0.0  # rho

        values = -torch.log(
            0.01 +
            self.background_rate * self.gaussian_2d(inputs_bg, targets) +
            (0.99 - self.background_rate) * self.gaussian_2d(inputs, targets)
        )

        ## Used in variety loss (SGAN)
        if self.keep_batch_dim:
            values = values.reshape(pred_length, batch_size)
            return values.mean(dim=0)
        
        return torch.mean(values)

class L1Loss(torch.nn.Module):
    """L1 Loss 

    This Loss penalizes only the primary trajectories
    """
    def __init__(self, keep_batch_dim=False):
        super(L1Loss, self).__init__()
        self.loss = torch.nn.L1Loss(reduction='none')
        self.keep_batch_dim = keep_batch_dim

    def forward(self, inputs, targets, batch_split):
        ## Extract primary pedestrians
        # [pred_length, num_tracks, 2] --> [pred_length, batch_size, 2]
        targets = targets.transpose(0, 1)
        targets = targets[batch_split[:-1]]
        targets = targets.transpose(0, 1)
        # [pred_length, num_tracks, 5] --> [pred_length, batch_size, 5]
        inputs = inputs.transpose(0, 1)
        inputs = inputs[batch_split[:-1]]
        inputs = inputs.transpose(0, 1)

        loss = self.loss(inputs[:, :, :2], targets)

        ## Used in variety loss (SGAN)
        if self.keep_batch_dim:
            return loss.mean(dim=0).mean(dim=1)
        
        return torch.mean(loss)

class L2Loss(torch.nn.Module):
    """L2 Loss (deterministic version of PredictionLoss)

    This Loss penalizes only the primary trajectories
    """
    def __init__(self, keep_batch_dim=False):
        super(L2Loss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.keep_batch_dim = keep_batch_dim

    def forward(self, inputs, targets, batch_split):
        ## Extract primary pedestrians
        # [pred_length, num_tracks, 2] --> [pred_length, batch_size, 2]
        targets = targets.transpose(0, 1)
        targets = targets[batch_split[:-1]]
        targets = targets.transpose(0, 1)
        # [pred_length, num_tracks, 5] --> [pred_length, batch_size, 5]
        inputs = inputs.transpose(0, 1)
        inputs = inputs[batch_split[:-1]]
        inputs = inputs.transpose(0, 1)

        loss = self.loss(inputs[:, :, :2], targets)

        ## Used in variety loss (SGAN)
        if self.keep_batch_dim:
            return loss.mean(dim=0).mean(dim=1)
        
        return torch.mean(loss)

class VarietyLoss(torch.nn.Module):
    def __init__(self, criterion, pred_length, loss_multiplier):
        super(VarietyLoss, self).__init__()
        self.criterion = criterion
        self.pred_length = pred_length
        self.loss_multiplier = loss_multiplier

    def forward(self, inputs, targets, batch_split):
        """ Variety loss calculation as proposed in SGAN

        Parameters
        ----------
        inputs : List of length k
            Each element of the list is Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        target : Tensor [pred_length, num_tracks, 2]
            Groundtruth sequence of primary pedestrians of each scene
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the primary tracks of each scene
        TODO: add following param
        Returns
        -------
        loss : Tensor [1,]
            variety loss
        """

        iterative_loss = [] 
        for sample in inputs:
            sample_loss = self.criterion(sample[-self.pred_length:], targets, batch_split) * self.loss_multiplier
            iterative_loss.append(sample_loss)

        loss = torch.stack(iterative_loss)
        loss = torch.min(loss, dim=0)[0]
        loss = torch.sum(loss)
        return loss

class ReconstructionLoss(torch.nn.Module):
    def __init__(self, criterion, pred_length, loss_multiplier, batch_size, num_modes):
        super(ReconstructionLoss, self).__init__()
        self.criterion = criterion
        self.pred_length = pred_length
        self.loss_multiplier = loss_multiplier
        self.batch_size = batch_size
        self.num_modes = num_modes

    def forward(self, inputs, targets, batch_split):
        """ Reconstruction loss

        Parameters
        ----------
        inputs : List of length k
            Each element of the list is : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        target : Tensor [pred_length, num_tracks, 2]
            Groundtruth sequence of primary pedestrians of each scene
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the primary tracks of each scene

        Return
        -------
        loss : Tensor [1,]
            recontruction loss
        """

        loss = 0
        for sample in inputs:
            loss += self.criterion(sample[-self.pred_length:], targets, batch_split) * self.batch_size * self.loss_multiplier / self.num_modes
        return loss