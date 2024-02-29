import torch
import math
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def kl_div(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def xavier_truncated_normal(size, no_average=False):
    """Samples from a truncated normal where the standard deviation is automatically chosen based on size."""
    if isinstance(size, int):
        size = (size,)

    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2

    return nn.init.trunc_normal_(torch.empty(size), std=(1 / n_avg) ** 0.5)


def flatten(list_of_lists):
    """Turn a list of lists (or any iterable) into a flattened list."""
    return [item for sublist in list_of_lists for item in sublist]


def find_closest(sample_times, true_times, equality_allowed=False, effective_zero=0.0):
    """For each value in sample_times, find the values and associated indices in true_times that are
    closest and strictly less than. Both times can be in random orders.

    Arguments:
        sample_times {torch.FloatTensor} -- Contains times that we want to find values closest but not over them in true_times
        true_times {torch.FloatTensor} -- Will take the closest times from here compared to sample_times
        effective_zero {float, torch.FloatTensor} -- If both a true event time and a sample time happen to be this value exactly, then it will be included in the mask. Useful when wanting to start integration

    Returns:
        dict -- Contains the closest values and corresponding indices from true_times.
    """
    # Pad true events with zeros (if a value in t is smaller than all of true_times, then we have it compared to time=0)
    if true_times.shape[-1] == 0:
        padded_true_times = torch.zeros(*true_times.shape[:-1], 1, device=true_times.device, dtype=torch.float32)
    else:
        padded_true_times = torch.cat((true_times[..., [0] ] *0, true_times), dim=-1)

    # Format true_times to have all values compared against all values of t
    size = padded_true_times.shape
    expanded_true_times = padded_true_times.unsqueeze(-1).expand(*size, sample_times.shape[-1])
    expanded_true_times = expanded_true_times.permute(*list(range(len(size ) -1)), -1, -2)

    # Find out which true event times happened after which times in t, then mask them out
    # if equality_allowed:
    #     mask = (expanded_true_times <= sample_times.unsqueeze(-1))
    # else:
    mask = (expanded_true_times < sample_times.unsqueeze(-1))
    if isinstance(effective_zero, float):
        mask = mask | ((expanded_true_times == 0.0) & (sample_times.unsqueeze(-1) == 0.0))
    else:
        assert((len(effective_zero.shape) == 1) and
                    (effective_zero.shape[0] == true_times.shape[0]))  # Single (batch) dimension
        print(effective_zero.shape, expanded_true_times.shape, sample_times.shape)
        raise NotImplementedError
    adjusted_expanded_true_times = torch.where(mask, expanded_true_times, -expanded_true_times * float('inf'))

    # Find the largest, unmasked values. These are the closest true event times that happened prior to the times in t.
    closest_values, closest_indices = adjusted_expanded_true_times.max(dim=-1)
    # closest_values = torch.nan_to_num(closest_values, nan=0.0)  # cover edge case when sample_times == 0.0

    return {
        "closest_values": closest_values,
        "closest_indices": closest_indices,
    }


def dpp_plot_marginal_kernel(DPP, save_path=''):
    '''Self-defined function to avoid numerical errors in computing eigenvalues of L'''
    if DPP.L_gram_factor:
        phi, phi_T = torch.tensor(DPP.L_gram_factor), torch.tensor(np.transpose(DPP.L_gram_factor))
        L = torch.matmul(phi_T, phi)
        num_item = L.shape[-1]
        I = torch.eye(num_item)
        K = I - torch.linalg.inv(L + I)
    else:
        K = DPP.K

    fig, ax = plt.subplots(1, 1)
    heatmap = ax.pcolor(K.numpy(), cmap='jet', vmin=-0.3, vmax=1)
    ax.set_aspect('equal')

    ticks = np.arange(num_item)
    ticks_label = [r'${}$'.format(tic) for tic in ticks]

    ax.xaxis.tick_top()
    ax.set_xticks(ticks + 0.5, minor=False)

    ax.invert_yaxis()
    ax.set_yticks(ticks + 0.5, minor=False)

    ax.set_xticklabels(ticks_label, minor=False)
    ax.set_yticklabels(ticks_label, minor=False)

    plt.colorbar(heatmap)
    plt.show()

    if save_path:
        plt.savefig(save_path)
    return


def flatten(list_of_lists):
    """Turn a list of lists (or any iterable) into a flattened list."""
    return [item for sublist in list_of_lists for item in sublist]