"""Loss used in the VAE model"""

import math
import torch

class KLDLoss(torch.nn.Module):
    """Kullback-Leibler divergence Loss

    """
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, inputs, targets = None):
        """
        Forward path

        Parameters:
        -----------
        inputs : Tensor [batch_size, 2*latent_dim]
            Tensor containing multivariate distribution mean and logarithmic variance
        targets : Tensor [batch_size, 2*latent_dim] TODO: Not implemented yet.
            Tensor containing target multivariate distribution mean and logarithmic variance
            Default: standard normal distribution (zero mean and unit variance)

        """
        if targets is None:
            z_mu, z_log_var = torch.split(inputs, split_size_or_sections=inputs.size(1)//2, dim=1)
            latent_loss = -0.5 * torch.sum(1.0 + z_log_var - torch.square(z_mu) - torch.exp(z_log_var), dim=1)
            return torch.mean(latent_loss)
        else:
            raise NotImplementedError

    

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
