import torch
import torch.nn as nn
import torch.nn.functional as F
from .ppmodel import PPModel
from .decoder import HawkesDecoder, RMTPPDecoder

from abc import abstractmethod
from dppy.finite_dpps import FiniteDPP
from .utils import *
from tqdm import tqdm

class CondIndepSets(PPModel):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(
                decoder,
                num_channels,
                channel_embedding,
                dyn_dom_buffer
        )

    @staticmethod
    def log_likelihood(return_dict, target_marks, right_window, left_window=0.0, mask=None, normalize_by_window=False, normalize_by_events=False):
        '''
                Computes per-batch log-likelihood from the results of a forward pass (that included a set of sample points).

                :param return_dict: dict_keys(['state_dict', 'intensities', 'sample_intensities'])
                :param target_marks: torch.Size([batch_size, num_events, num_channels])

                :return:
                '''
        assert ("intensities" in return_dict)
        assert ("sample_intensities" in return_dict and "total_intensity" in return_dict["sample_intensities"])
        assert ("total_intensity" in return_dict["intensities"] and "item_probability" in return_dict["intensities"]
                and "item_logits" in return_dict["intensities"])

        # mask: torch.Size([batch_size, num_events])
        if mask is None:
            mask = True
        else:
            mask = mask.bool()

        total_intensity = return_dict["intensities"]["total_intensity"].squeeze(-1)
        log_total_intensity = torch.log(torch.where(mask, total_intensity, torch.ones_like(total_intensity)))  # torch.Size([batch_size, num_events])
        item_logits = return_dict["intensities"]["item_logits"]

        log_item_prob = F.logsigmoid(torch.where(target_marks > 0, item_logits, -item_logits))  # log p(y=1) = log sigmoid(x), log p(y=0) = log sigmoid(-x) for x=item_logits
        log_item_prob_by_events = torch.sum(torch.where(mask.unsqueeze(-1).repeat(1, 1, target_marks.shape[-1]),
                                                            log_item_prob, torch.zeros_like(log_item_prob)), dim=-1)

        positive_samples_mark = torch.sum(log_item_prob_by_events, dim=-1, keepdim=True)  # allow empty sets
        positive_samples_pp = torch.sum(log_total_intensity, dim=-1, keepdim=True)
        negative_samples = (right_window - left_window) * return_dict["sample_intensities"]["total_intensity"].squeeze(-1).mean(dim=-1, keepdim=True)
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


    def sample_single_set(self, input_dict, idx, mark_mask=None):
        item_probs = (input_dict["item_probability"][:, [idx], :])  # torch.Size([1, 1, num_channel])
        if mark_mask != None:
            item_probs = torch.where(mark_mask[:, [idx], :], item_probs, 0)
        new_set = torch.bernoulli(item_probs)
        return new_set


    def sample_multiple_set(self, input_dict, batch_idx, event_idx, mark_mask=None):
        '''
        :param input_dict['item_probability']: torch.Size([num_samples in batch sampling, num_events, num_channel])
        :param len(batch_idx) = len(event_idx) = num_samples

        return sampled sets: torch.Size([num_samples, num_channel])
        '''
        item_probs = input_dict['item_probability'][batch_idx, event_idx, :]  # torch.Size([num_samples, num_channel])
        if mark_mask != None:
            item_probs = torch.where(mark_mask[batch_idx, event_idx, :], item_probs, 0)  # zero out restricted items
        new_sets = torch.bernoulli(item_probs)  # torch.Size([num_samples, num_channel])
        return new_sets


    @abstractmethod
    def get_total_intensity_and_items(self, h_t):
        raise NotImplementedError("Must override get_total_intensity_and_items")



class StaticCondIndepSets(CondIndepSets):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(
                decoder,
                num_channels,
                channel_embedding,
                dyn_dom_buffer
        )
        self.recurrent_hidden_size = self.decoder.recurrent_hidden_size
        self.hidden_to_total_intensity = nn.Linear(self.recurrent_hidden_size, 1, bias=True)
        self.item_logits = nn.Parameter(torch.zeros(self.num_channels))

    def get_total_intensity_and_items(self, h_t):
        if isinstance(self.decoder, HawkesDecoder):
            total_intensity = F.softplus(self.hidden_to_total_intensity(h_t))
        elif isinstance(self.decoder, RMTPPDecoder):
            total_intensity = torch.exp(self.hidden_to_total_intensity(h_t))
        batch_size, num_events, _ = h_t.shape
        item_logits = self.item_logits.expand(batch_size, num_events, -1)
        item_probability = torch.sigmoid(item_logits)
        return {"total_intensity": total_intensity,
                "item_logits": item_logits,
                "item_probability": item_probability}



class DynamicCondIndepSets(CondIndepSets):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer, num_layer_item=1):
        super().__init__(
                decoder,
                num_channels,
                channel_embedding,
                dyn_dom_buffer
        )
        self.recurrent_hidden_size = self.decoder.recurrent_hidden_size
        self.half_h_size = self.recurrent_hidden_size//2
        self.hidden_to_total_intensity = nn.Linear(self.half_h_size, 1, bias=True)
        if num_layer_item == 1:
            self.hidden_to_item_logits = nn.Linear(self.half_h_size, self.num_channels, bias=True)
        else:
            hidden_to_item_layers = []
            for _ in range(num_layer_item - 1):
                hidden_to_item_layers += [(nn.Linear(self.half_h_size, self.half_h_size, bias=True))]
            hidden_to_item_layers += [nn.Linear(self.half_h_size, self.num_channels, bias=True)]
            self.hidden_to_item_logits = nn.Sequential(*hidden_to_item_layers)


    def get_total_intensity_and_items(self, h_t):
        h_t_lambda, h_t_m = torch.split(h_t, [self.half_h_size, self.half_h_size], dim=-1)
        if isinstance(self.decoder, HawkesDecoder):
            total_intensity = F.softplus(self.hidden_to_total_intensity(h_t_lambda))
        elif isinstance(self.decoder, RMTPPDecoder):
            total_intensity = torch.exp(self.hidden_to_total_intensity(h_t_lambda))
        item_logits = self.hidden_to_item_logits(h_t_m)  # torch.Size([batch_size, events, channel])
        item_probability = torch.sigmoid(item_logits)
        return {"total_intensity": total_intensity,
                "item_logits": item_logits,
                "item_probability": item_probability}



class DPPSets(PPModel):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(
                decoder,
                num_channels,
                channel_embedding,
                dyn_dom_buffer
        )

    @staticmethod
    def log_likelihood(return_dict, target_marks, right_window, left_window=0.0, mask=None, normalize_by_window=False,
                       normalize_by_events=False, gamma = 0.):
        assert ("intensities" in return_dict)
        assert ("sample_intensities" in return_dict and "total_intensity" in return_dict["sample_intensities"])
        assert ("total_intensity" in return_dict["intensities"] and "feature_vectors" in return_dict["intensities"]
                and "feature_vectors_T" in return_dict["intensities"])

        if mask is None:
            mask = True
        else:
            mask = mask.bool()  # batch_size * num_events

        total_intensity = return_dict["intensities"]["total_intensity"].squeeze(-1)
        log_total_intensity = torch.log(torch.where(mask, total_intensity, torch.ones_like(total_intensity)))

        phi, phi_T = return_dict["intensities"]["feature_vectors"], return_dict["intensities"]["feature_vectors_T"]
        L = torch.matmul(phi_T, phi)
        I_B, I_B_bar = torch.diag_embed(target_marks), torch.diag_embed(1 - target_marks)
        if L.get_device() != -1:
            I = torch.eye(L.shape[-1]).cuda(torch.cuda.current_device())
        else:
            I = torch.eye(L.shape[-1])
        K = I - torch.linalg.inv(L + I)

        negative_samples = (right_window - left_window) * return_dict["sample_intensities"]["total_intensity"].squeeze(-1).mean(dim=-1, keepdim=True)
        log_event_prob = torch.log(torch.clamp(torch.det(torch.matmul(I_B, K) + torch.matmul(I_B_bar, I - K)), min=1e-20))  # batch_size * num_events
        positive_samples_mark = torch.sum(torch.where(mask, log_event_prob, torch.zeros_like(log_event_prob)), dim=-1, keepdim=True)

        positive_samples_pp = torch.sum(log_total_intensity, dim=-1, keepdim=True)
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

    def sample_single_set(self, input_dict, idx, plot_marginal_kernel=False):
        if input_dict["feature_vectors"].get_device() != -1:
            DPP = FiniteDPP('likelihood', **{
                'L_gram_factor': input_dict["feature_vectors"][:, [idx], :, :].squeeze().detach().cpu().numpy()})
        else:
            DPP = FiniteDPP('likelihood', **{'L_gram_factor': input_dict["feature_vectors"][:, [idx], :, :].squeeze().detach().numpy()})

        if plot_marginal_kernel:
            dpp_plot_marginal_kernel(DPP)
        new_mark = DPP.sample_exact()  # allow empty set
        return new_mark


    def sample_multiple_set(self, input_dict, batch_idx, event_idx, mark_mask=None):
        if mark_mask != None:
            raise NotImplementedError
        else:
            new_marks = []
            for i in range(len(batch_idx)):
                if input_dict["feature_vectors"].get_device() != -1:
                    DPP = FiniteDPP('likelihood', **{'L_gram_factor': input_dict["feature_vectors"][batch_idx[i], event_idx[i], :, :].squeeze().detach().cpu().numpy()})
                else:
                    DPP = FiniteDPP('likelihood', **{'L_gram_factor': input_dict["feature_vectors"][batch_idx[i], event_idx[i], :, :].squeeze().detach().numpy()})
                new_mark = DPP.sample_exact()
                new_mark_multi_hot = torch.eye(self.num_channels)[new_mark].sum(dim=0).float().unsqueeze(0).to(batch_idx.device)
                new_marks.append(new_mark_multi_hot)
            new_marks = torch.cat(new_marks, dim=0)
        return new_marks

    @abstractmethod
    def get_total_intensity_and_items(self, h_t):
        raise NotImplementedError("Must override get_total_intensity_and_items")



class StaticDPPSets(DPPSets):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(
                decoder,
                num_channels,
                channel_embedding,
                dyn_dom_buffer
        )
        self.recurrent_hidden_size = self.decoder.recurrent_hidden_size
        self.hidden_to_total_intensity = nn.Linear(self.recurrent_hidden_size, 1, bias=True)
        self.item_logits = nn.Parameter(xavier_truncated_normal(size=(self.num_channels), no_average=True))

    def get_total_intensity_and_items(self, h_t):
        if isinstance(self.decoder, HawkesDecoder):
            total_intensity = F.softplus(self.hidden_to_total_intensity(h_t))
        elif isinstance(self.decoder, RMTPPDecoder):
            total_intensity = torch.exp(self.hidden_to_total_intensity(h_t))
        batch_size, num_events, _ = h_t.shape
        item_logits = self.item_logits.expand(batch_size, num_events, -1)
        item_quality = torch.exp(item_logits)
        phi_T = torch.matmul(torch.diag_embed(item_quality), F.normalize(self.channel_embedding.weight.data,
                                                                         dim=-1))  # torch.Size([batch_size, num_events, M, d])
        return {"total_intensity": total_intensity,
                "feature_vectors": torch.transpose(phi_T, -1, -2),
                "feature_vectors_T": phi_T}



class DynamicDPPSets(DPPSets):
    def __init__(self, decoder, num_channels, channel_embedding, dyn_dom_buffer):
        super().__init__(
            decoder,
            num_channels,
            channel_embedding,
            dyn_dom_buffer
        )
        self.recurrent_hidden_size = self.decoder.recurrent_hidden_size
        self.half_h_size = self.recurrent_hidden_size // 2
        self.hidden_to_total_intensity = nn.Linear(self.half_h_size, 1, bias=True)
        self.hidden_to_item_logits = nn.Linear(self.half_h_size, self.num_channels, bias=True)
        
    def get_total_intensity_and_items(self, h_t):
        h_t_lambda, h_t_m = torch.split(h_t, [self.half_h_size, self.half_h_size], dim=-1)
        if isinstance(self.decoder, HawkesDecoder):
            total_intensity = F.softplus(self.hidden_to_total_intensity(h_t_lambda))
        elif isinstance(self.decoder, RMTPPDecoder):
            total_intensity = torch.exp(self.hidden_to_total_intensity(h_t_lambda))
        item_logits = self.hidden_to_item_logits(h_t_m)  # torch.Size([batch_size, events, channel])
        item_quality = torch.exp(item_logits)
        phi_T = torch.matmul(torch.diag_embed(item_quality), F.normalize(self.channel_embedding.weight.data,
                                                                         dim=-1))  # torch.Size([batch_size, num_events, M, d])
        return {"total_intensity": total_intensity,
                "feature_vectors": torch.transpose(phi_T, -1, -2),
                "feature_vectors_T": phi_T}
