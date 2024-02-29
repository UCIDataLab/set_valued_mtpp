import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from .utils import *
from .set_assumption import CondIndepSets


class PoissonStaticCI(CondIndepSets):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(decoder, num_channels, channel_embedding, dyn_dom_buffer)
        self.num_channels = num_channels
        self.rate = nn.Parameter(torch.zeros(1), requires_grad=False)  # rate parameter for total intensity
        self.item_logits = nn.Parameter(torch.zeros(self.num_channels), requires_grad=False)


    def forward(self, marks, timestamps, sample_timestamps=None):
        '''
        marks: torch.Size([batch_size, events, channel])
        timestamps: torch.Size([batch_size, events])
        sample_timestamps: torch.Size([batch_size, 150])
        '''
        results = {}
        results['intensities'] = self.get_intensity(marks, timestamps, timestamps, None)
        if sample_timestamps is not None:
            results['sample_intensities'] = self.get_intensity(marks, timestamps, sample_timestamps, None)
        results["state_dict"] = {"state_values": marks, "state_times": timestamps}
        return results


    def get_intensity(self, state_values, state_times, timestamps, marks=None):
        sample_timestamps = timestamps
        batch_size, num_events = sample_timestamps.shape

        total_intensity = self.rate.expand(batch_size, num_events, -1).to(sample_timestamps.device)
        item_logits = self.item_logits.expand(batch_size, num_events, -1).to(sample_timestamps.device)

        return {"total_intensity": total_intensity,
              "item_logits": item_logits,
              "item_probability": torch.sigmoid(item_logits)}


    def get_poisson_statistics(self, dataloader):
        rate, item_prob = 0, torch.zeros(self.num_channels)
        data_len = len(dataloader)  # drop_last=True for train_dataloader
        for i, batch in enumerate(dataloader):
            mask, T = batch["padding_mask"].bool(), batch["T"]
            target_marks, target_timestamps = batch["target_marks"], batch["target_times"]

            total_num_events = torch.sum(mask)
            rate += total_num_events / torch.sum(T) / data_len
            item_prob += (torch.sum(torch.where(mask.unsqueeze(-1).repeat(1, 1, target_marks.shape[-1]),
                                                target_marks, 0), (0,1))/ total_num_events) / data_len
        self.rate.data = torch.Tensor([rate])
        self.item_logits.data = torch.logit(item_prob)


    @staticmethod
    def log_likelihood(return_dict, target_marks, right_window, left_window=0.0, mask=None, normalize_by_window=False,
                       normalize_by_events=False, gamma=0.):
        '''
        Computes per-batch log-likelihood from the results of a forward pass (that included a set of sample points).

        :param return_dict: dict_keys(['state_dict', 'intensities', 'sample_intensities'])
        :param target_marks: torch.Size([batch_size, num_events, num_channels])

        :return: log likelihood of each components
        '''
        assert ("intensities" in return_dict)
        assert ("sample_intensities" in return_dict and "total_intensity" in return_dict["sample_intensities"])
        # item_probability are required for the conditional independence assumption
        assert ("total_intensity" in return_dict["intensities"] and "item_probability" in return_dict["intensities"]
                and "item_logits" in return_dict["intensities"])

        # mask: torch.Size([batch_size, num_events])
        if mask is None:
            mask = True
        else:
            mask = mask.bool()

        total_intensity = return_dict["intensities"]["total_intensity"].squeeze(-1)
        log_total_intensity = torch.log(torch.where(mask, total_intensity+1e-20, torch.ones_like(total_intensity)))  # torch.Size([batch_size, num_events])

        item_logits = return_dict["intensities"]["item_logits"]

        log_item_prob = F.logsigmoid(torch.where(target_marks > 0, item_logits,
                                                 -item_logits))  # log p(y=1) = log sigmoid(x), log p(y=0) = log sigmoid(-x) for x=item_logits
        log_item_prob_by_events = torch.sum(torch.where(mask.unsqueeze(-1).repeat(1, 1, target_marks.shape[-1]),
                                                            log_item_prob, torch.zeros_like(log_item_prob)), dim=-1)

        positive_samples_mark = torch.sum(log_item_prob_by_events, dim=-1, keepdim=True)  # allow empty sets
        positive_samples_pp = torch.sum(log_total_intensity, dim=-1, keepdim=True)

        # positive_samples = torch.sum(log_item_prob - log_norm_term + log_total_intensity, dim=-1, keepdim=True)  # torch.Size([32, 25]) -> 32 * 1
        negative_samples = (right_window - left_window) * return_dict["sample_intensities"]["total_intensity"].squeeze(
            -1).mean(-1)  # torch.Size([batch_size, 1])
        timing_contribution = positive_samples_pp - negative_samples

        ll_results = {
            "log_likelihood": (positive_samples_mark + timing_contribution).mean(),
            "positive_contribution_marks": positive_samples_mark.mean(),
            "positive_contribution_pp": positive_samples_pp.mean(),
            "negative_contribution": negative_samples.mean(),
            "timing_contribution": timing_contribution.mean()
        }

        if normalize_by_events:  # sum_{seq}(ll_mark)/total_num_events
            batch_N = mask.sum(dim=-1)
            ll_results["positive_contribution_marks_norm"] = positive_samples_mark.sum() / batch_N.sum()
        return ll_results


    def get_param_groups(self):
        """Returns iterable of dictionaries specifying parameter groups.
        The first dictionary in the return value contains parameters that will be subject to weight decay.
        The second dictionary in the return value contains parameters that will not be subject to weight decay.

        Returns:
            (param_group, param_groups) -- Tuple containing sets of parameters, one of which has weight decay enabled, one of which has it disabled.
        """
        NORMS = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        )

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        for module_ in self.modules():
            # Doesn't make sense to decay weights for a LayerNorm, BatchNorm, etc.
            if isinstance(module_, NORMS):
                no_weight_decay_params['params'].extend([
                    p for p in module_._parameters.values() if p is not None
                ])
            else:
                # Also doesn't make sense to decay biases.
                weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n != 'bias'
                ])
                no_weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n == 'bias'
                ])

        return weight_decay_params, no_weight_decay_params