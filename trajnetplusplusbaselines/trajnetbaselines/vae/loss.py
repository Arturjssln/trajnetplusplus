"""Loss used in the VAE model"""

import torch

def kld_loss(z):
    """
    Input:
    - z: Tensor of shape (????) corresponding to the distribution of latent variable

    Output:
    - loss:
    """
    raise NotImplementedError
    loss = torch.zeros(z.size())
    return loss