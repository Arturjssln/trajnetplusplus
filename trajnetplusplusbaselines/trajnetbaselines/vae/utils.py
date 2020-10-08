import math
import random

import numpy as np
import torch

def shift(xy, center):
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[np.newaxis, np.newaxis, :]
    return xy

def theta_rotation(xy, theta):
    # theta = random.random() * 2.0 * math.pi
    ct = np.cos(theta)
    st = np.sin(theta)

    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)

def random_rotation(xy, goals=None):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])
    if goals is None:
        return np.einsum('ptc,ci->pti', xy, r)
    return np.einsum('ptc,ci->pti', xy, r), np.einsum('tc,ci->ti', goals, r)

def center_scene(xy, obs_length=9, ped_id=0, goals=None):
    if goals is not None:
        goals = goals[np.newaxis, :, :]
    ## Center
    center = xy[obs_length-1, ped_id] ## Last Observation
    xy = shift(xy, center)
    if goals is not None:
        goals = shift(goals, center)
    ## Rotate
    last_obs = xy[obs_length-1, ped_id]
    second_last_obs = xy[obs_length-2, ped_id]
    diff = np.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = np.arctan2(diff[1], diff[0])
    rotation = -thet + np.pi/2
    xy = theta_rotation(xy, rotation)
    if goals is not None:
        goals = theta_rotation(goals, rotation)
        return xy, rotation, center, goals[0]
    return xy, rotation, center

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

def sample_multivariate_distribution(mean, var, n_samples=None):
    """
    Draw random samples from a multivariate normal distribution 

    Parameters
    ----------
    mean : Tensor [dim]
        Mean of the multivariate distribution
    var : Tensor [dim]
        Diagonal coefficient of the covariance matrix
    n_samples : int
        Number of samples to draw. 
        When not specified, a single sample is returned.

    Returns
    -------
    samples : Tensor [n_samples, dim] or [dim]
        The drawn samples, of size [n_samples, dim] if n_samples was provided. 
        If not, the size is [dim]

    """
    cov_matrix = np.diag(var.numpy())
    samples = np.random.multivariate_normal(mean.numpy(), cov_matrix, size=n_samples)
    return torch.Tensor(samples)