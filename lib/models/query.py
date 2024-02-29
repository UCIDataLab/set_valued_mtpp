import torch
import torch.nn.functional as F
import math

from tqdm import tqdm
from abc import abstractmethod
from lib.models.utils import flatten

ADAPT_DOM_RATE = True

class Query:
    @abstractmethod
    def naive_estimate(self, model, num_sample_seq):
        pass

    @abstractmethod
    def estimate(self, model, num_sample_seq, num_int_samples):
        pass

    @abstractmethod
    def proposal_dist_sample(self, model, num_samples=1):
        pass

class TemporalMarkQuery(Query):

    def __init__(self, time_boundaries, mark_restrictions, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
        if isinstance(time_boundaries, list):
            time_boundaries = torch.FloatTensor(time_boundaries)

        self.time_boundaries = time_boundaries.to(device)
        self.mark_restrictions = []    # list of list of restricted marks, if k is in one of these lists, then events with marks=k are not allowed in that respected time span
        for m in mark_restrictions:
            if isinstance(m, int):
                m = torch.LongTensor([m])
            elif isinstance(m, list):
                m = torch.LongTensor(m)
            else:
                assert(isinstance(m, torch.LongTensor))
            self.mark_restrictions.append(m.to(device))

        self.max_time = max(time_boundaries)
        self.num_times = len(time_boundaries)
        assert(time_boundaries[0] > 0)
        assert((time_boundaries[:-1] < time_boundaries[1:]).all())    # strictly increasing
        assert(len(time_boundaries) == len(mark_restrictions))

        self.device = device
        self.batch_size = batch_size
        self.restricted_positions = torch.BoolTensor([len(m) > 0 for m in self.mark_restrictions]).to(device)
        self.use_tqdm = use_tqdm
        self.proposal_batch_size = proposal_batch_size

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seq, conditional_times=None, conditional_marks=None):
        res = 0.0    # for a sample to count as 1 in the average, it must respect _every_ mark restriction
        if (conditional_times is None) or (conditional_times.numel() == 0):
            offset = 0.0
        else:
            offset = conditional_times.max()

        all_times, all_marks, _ = model.batch_sample_points(
            T=self.max_time+offset,
            left_window=0+offset,
            timestamps=conditional_times,
            marks=conditional_marks,
            # mark_mask=None,
            num_samples=num_sample_seq,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        all_times = flatten([times.unbind(dim=0) for times in all_times])
        all_marks = flatten([marks.unbind(dim=0) for marks in all_marks])

        # times: (num_events), marks: (num_events, num_channel)
        for times, marks in tqdm(zip(all_times, all_marks), disable=not self.use_tqdm):
            if conditional_times is not None:
                times = times[conditional_times.numel():] - offset
                marks = marks[conditional_times.numel():]

            for i in range(self.num_times):
                if i == 0:
                    a, b = 0.0, self.time_boundaries[0]
                else:
                    a, b = self.time_boundaries[i-1], self.time_boundaries[i]

                restricted_marks = torch.eye(model.num_channels)[self.mark_restrictions[i]].sum(dim=0).to(self.device)
                if (marks[(a < times) & (times <= b)] * (restricted_marks == 1)).any():
                    break
            else:    # Executes if for loop did not break
                res += 1. / num_sample_seq
        return res


    @torch.no_grad()
    def proposal_dist_sample(self, model, conditional_times=None, conditional_marks=None, offset=None, num_samples=1):
        times, marks = conditional_times, conditional_marks
        if offset is None:
            if (conditional_times is None) or (conditional_times.numel() == 0):
                offset = 0.0
            else:
                offset = conditional_times.max()

        mark_mask = torch.ones((len(self.time_boundaries) + 1, model.num_channels,), dtype=torch.float32).to(self.device)  # +1 to have an unrestricted final mask
        for i in range(len(self.time_boundaries)):
            mark_mask[i, self.mark_restrictions[i]] = 0.0
        mask_dict = {
            "temporal_mark_restrictions": mark_mask,
            "time_boundaries": self.time_boundaries + offset,
        }


        all_times, all_marks, all_states = model.batch_sample_points(
            T=self.time_boundaries.max() + offset,
            left_window=offset,
            timestamps=times,
            marks=marks,
            num_samples=num_samples,
            mask_dict=mask_dict,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
        )
        return all_times, all_marks, all_states, mask_dict


    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        # If true, then we will need to integrate for events in that position
        time_spans = self.time_boundaries - F.pad(self.time_boundaries[:-1].unsqueeze(0), (1, 0), 'constant', 0.0).squeeze(0)  # Equivalent to: np.ediff1d(self.time_boundaries, to_begin=self.time_boundaries[0])
        time_norm = time_spans[self.restricted_positions].sum()  # Used to scale how many integration sample points each interval uses
        ests = []
        est_lower_bound, est_upper_bound = 0.0, 0.0
        if (conditional_times is None) or (conditional_times.numel() == 0):
            offset = 0.0
        else:
            offset = conditional_times.max().item()  # +1e-32

        batch_sizes = [min(self.proposal_batch_size, num_sample_seqs - i * self.proposal_batch_size) for i in
                       range(math.ceil(num_sample_seqs / self.proposal_batch_size))]  # list of batch_sizes that sums up to num_sample_seqs
        for batch_size in batch_sizes:
            all_times, all_marks, all_states, mask_dict = self.proposal_dist_sample(model, conditional_times=conditional_times,
                                                                         conditional_marks=conditional_marks,
                                                                         num_samples=batch_size)

            for times, marks, states in tqdm(zip(all_times, all_marks, all_states), disable=not self.use_tqdm):
                # times, marks = self.proposal_dist_sample(model, conditional_times=conditional_times, conditional_marks=conditional_marks)
                total_int, lower_int, upper_int = 0.0, 0.0, 0.0
                # print(conditional_times.shape, conditional_times.numel(), times.shape, marks.shape, states.shape)

                for i in range(len(time_spans)):
                    if not self.restricted_positions[i]:
                        continue

                    if i == 0:
                        a, b = 0.0, self.time_boundaries[0].item()
                    else:
                        a, b = self.time_boundaries[i - 1].item(), self.time_boundaries[i].item()
                    a = a + offset
                    b = b + offset
                    single_res = model.compensator(
                        a,
                        b,
                        conditional_times=times,
                        conditional_marks=marks,
                        conditional_states=states,
                        num_int_pts=max(int(num_int_samples * time_spans[i] / time_norm), 2), # At least two points are needed to integrate
                        calculate_bounds=calculate_bounds,
                        mark_mask = mask_dict["temporal_mark_restrictions"][i].bool(), # 1 allowed, 0 restricted
                    )

                    if calculate_bounds:
                        raise NotImplementedError

                    total_int += single_res["integral"]

                ests.append(torch.exp(-total_int))
                if calculate_bounds:
                    raise NotImplementedError

        ests = torch.cat(ests, dim=0)
        assert (ests.numel() == num_sample_seqs)
        est = ests.mean()
        results = {
            "est": est,
            "naive_var": est - est ** 2,
            "is_var": (est - ests).pow(2).mean(),  # sum((e-est)**2 for e in ests) / num_sample_seqs,
        }
        results["rel_eff"] = results["naive_var"] / results["is_var"]
        return results



class UnbiasedHittingTimeQuery(TemporalMarkQuery):

    def __init__(self, up_to, hitting_marks, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
        assert(isinstance(up_to, (float, int)) and up_to > 0)
        assert(isinstance(hitting_marks, (int, list)))
        super().__init__(
            time_boundaries=[up_to],
            mark_restrictions=[hitting_marks],
            batch_size=batch_size,
            device=device,
            use_tqdm=use_tqdm,
            proposal_batch_size=proposal_batch_size,
        )

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        return 1 - super().naive_estimate(model, num_sample_seqs, conditional_times=conditional_times, conditional_marks=conditional_marks)

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None, calculate_bounds=False):
        result = super().estimate(
            model,
            num_sample_seqs,
            num_int_samples,
            conditional_times=conditional_times,
            conditional_marks=conditional_marks,
            calculate_bounds=calculate_bounds,
        )
        result["est"] = 1 - result["est"]
        if calculate_bounds:
            result["lower_est"], result["upper_est"] = 1 - result["upper_est"], 1 - result["lower_est"]  # Need to calculate the complement and swap upper and lower bounds
        return result


class AbeforeBQuery(TemporalMarkQuery):
    def __init__(self, up_to, A, B, batch_size=256, device=torch.device('cpu'), use_tqdm=True, proposal_batch_size=1024):
        assert(isinstance(up_to, (float, int)) and up_to > 0)
        self.A = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.long).to(device)
        self.B = B if isinstance(B, torch.Tensor) else torch.tensor(B, dtype=torch.long).to(device)
        assert ((len(self.A.shape) == 1) and (self.A.numel() > 0))
        assert ((len(self.B.shape) == 1) and (self.B.numel() > 0))
        assert ((~torch.isin(self.A, self.B).any()).item())
        self.A_and_B = torch.cat((self.A, self.B), dim=0)

        super().__init__(
            time_boundaries=[up_to],
            mark_restrictions=[self.A_and_B],
            batch_size=batch_size,
            device=device,
            use_tqdm=use_tqdm,
            proposal_batch_size=proposal_batch_size,
        )

    @torch.no_grad()
    def naive_estimate(self, model, num_sample_seqs, conditional_times=None, conditional_marks=None):
        res = {}
        stop_marks = torch.eye(model.num_channels)[self.A_and_B].sum(dim=0).to(self.device)
        offset = 0 if conditional_times is None else conditional_times.max()
        all_times, all_marks, _ = model.batch_sample_points(
            left_window=offset,
            stop_marks=stop_marks,
            timestamps=conditional_times,
            marks=conditional_marks,
            mark_mask=None,
            num_samples=num_sample_seqs,
            adapt_dom_rate=ADAPT_DOM_RATE,
            proposal_batch_size=self.proposal_batch_size,
            T=self.time_boundaries[0] + offset,
        )
        all_marks = torch.cat(all_marks, dim=0)
        all_marks_prob = all_marks.mean(dim=-2)
        a_leq_b = all_marks_prob[self.A].cpu().item()
        b_leq_a = all_marks_prob[self.B].cpu().item()
        res['a_equals_b'] = ((all_marks * (stop_marks==1)).sum(dim=-1) == sum(stop_marks)).float().mean().cpu().item()
        res['a_before_b'] = a_leq_b - res['a_equals_b']
        res['b_before_a'] = b_leq_a - res['a_equals_b']
        res['no_a_or_b'] = 1 - a_leq_b - res['b_before_a']
        return res

    @torch.no_grad()
    def estimate(self, model, num_sample_seqs, num_int_samples, conditional_times=None, conditional_marks=None):
        offset = 0 if conditional_times is None else conditional_times.max()
        sampled_times, sampled_marks, sampled_states, mask_dict = self.proposal_dist_sample(model,
                                            conditional_times, conditional_marks, num_samples=num_sample_seqs)
        ts = torch.linspace(1e-20, self.time_boundaries[0], num_int_samples).to(self.device)

        all_a_equals_b, all_a_before_b, all_b_before_a, all_no_a_or_b = [], [], [], []
        for st, _, ss in zip(sampled_times, sampled_marks, sampled_states):  # list of tensors
            intensity_dict = model.get_intensity(
                state_values=ss,
                state_times=st,
                timestamps=ts.unsqueeze(0).expand(st.shape[0], -1) + offset,
                marks=None
            )
            vals = intensity_dict['total_intensity']  # (batch, num_int_samples, 1)
            item_probs = intensity_dict['item_probability']  # (batch, num_int_samples, num_channels)
            prob_a_and_b = item_probs[..., self.A] * item_probs[...,self.B]  # (batch, num_int_samples, 1)
            prob_a_or_b = item_probs[..., self.A] + item_probs[..., self.B] - prob_a_and_b  # (batch, num_int_samples, 1)
            comp = torch.exp(-F.pad(torch.cumulative_trapezoid((vals * prob_a_or_b).squeeze(-1), x=ts, dim=-1), (1, 0), 'constant', 0.0))  # (batch, num_int_samples)
            a_equals_b = torch.trapezoid((vals * prob_a_and_b).squeeze(-1) * comp, x=ts, dim=-1)  # (batch)
            a_before_b = torch.trapezoid((vals * (item_probs[..., self.A] - prob_a_and_b)).squeeze(-1) * comp, x=ts, dim=-1)
            b_before_a = torch.trapezoid((vals * (item_probs[..., self.B] - prob_a_and_b)).squeeze(-1) * comp, x=ts, dim=-1)
            all_no_a_or_b.append(1 - a_equals_b - a_before_b - b_before_a)
            all_a_equals_b.append(a_equals_b)
            all_a_before_b.append(a_before_b)
            all_b_before_a.append(b_before_a)

        all_a_equals_b = torch.cat(all_a_equals_b)
        all_a_before_b = torch.cat(all_a_before_b)
        all_b_before_a = torch.cat(all_b_before_a)
        all_no_a_or_b = torch.cat(all_no_a_or_b)
        a_before_b_avg = all_a_before_b.mean()
        res = {
            'a_equals_b': all_a_equals_b.mean().cpu().item(),
            'a_before_b': a_before_b_avg.cpu().item(),
            'b_before_a': all_b_before_a.mean().cpu().item(),
            'no_a_or_b': all_no_a_or_b.mean().cpu().item(),
            'naive_var': (a_before_b_avg * (1-a_before_b_avg)).cpu().item(),
            'is_var': (all_a_before_b - a_before_b_avg).pow(2).mean().cpu().item()
        }
        return res
