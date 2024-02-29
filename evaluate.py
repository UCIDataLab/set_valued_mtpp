import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
import math
from collections import defaultdict
from tqdm import tqdm
import time
import datetime
import random
import json
from lib.arguments import get_args
from lib.utils import *
from lib.models.utils import flatten
from lib.models.query import *


def _setup_hitting_time_query(args, batch, model, guarantee_mark=False, use_tqdm=False, num_item_to_query=1):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["target_times"], batch["target_marks"]
    length = times.numel()
    to_condition_on = min(length - 1, 5)
    last_time = times[:, to_condition_on-1].item()

    if args.checkpoint_path.startswith("../checkpoints/mooc_norm"):
        condition_marks = marks[:, :to_condition_on, ...]
        existing_marks = torch.nonzero(condition_marks.sum(dim=-2), as_tuple=True)[1]
        assert(len(existing_marks) >= 1)
        perm = torch.randperm(len(existing_marks))
        next_item = existing_marks[perm[0]].item()
    else:
        next_item = torch.randint(high=model.num_channels, size=(1,)).item()

    # ## generate queries based on num_item_to_query
    # if args.checkpoint_path.startswith("../checkpoints/mooc_norm"):
    #     condition_marks = marks[:, :to_condition_on, ...]
    #     existing_marks = torch.nonzero(condition_marks.sum(dim=-2), as_tuple=True)[1]
    #     assert (len(existing_marks) >= 1)
    #     if len(existing_marks) < num_item_to_query:
    #         existing_marks = list(range(model.num_channels))
    #     perm = torch.randperm(len(existing_marks))
    #     next_item = existing_marks[perm[:num_item_to_query]].tolist()
    # else:
    #     perm = torch.randperm(model.num_channels)
    #     next_item = perm[:num_item_to_query].tolist()

    up_to = max(min((times[:, to_condition_on].item() - last_time) * 10, 10.0), 1e-2)  # paper
    # up_to = max(min((times[:, to_condition_on].item() - last_time) * 10, 10.0, times[:, -1].item() - last_time), 1e-2)  # Dec 22
    # up_to = max(min((times[:, to_condition_on].item() - last_time) * 50, 50.0), 1e-2)  # Oct 7
    max_T = last_time + up_to


    remaining_times, remaining_marks = times[:, to_condition_on:], marks[:, to_condition_on:, ...]
    t_mask = remaining_times <= max_T
    remaining_times, remaining_marks = remaining_times[t_mask].unsqueeze(0), remaining_marks[t_mask].unsqueeze(0)

    if remaining_marks.sum(dim=-2).squeeze(0)[next_item].any() > 0:
        mark_obs, next_time = True, remaining_times[0, remaining_marks.argmax(dim=-2).squeeze(0)[next_item]].item() - last_time  # normalized

        # mark_obs = True
        # # A_idx = torch.nonzero(remaining_marks[:, :, next_item[0]], as_tuple=True)[1]
        # # B_idx = torch.nonzero(remaining_marks[:, :, next_item[1]], as_tuple=True)[1]
        # # idx = min(A_idx.item(), B_idx.item())
        # idx = min(torch.nonzero(remaining_marks[:, :, next_item], as_tuple=True)[1]).item()
        # next_time = remaining_times[0, idx].item() - last_time
    else:
        mark_obs, next_time = False, None

    times, marks = times[:, :to_condition_on], marks[:, :to_condition_on, ...]
    return times, marks, next_item, mark_obs, last_time, next_time, max_T, up_to



def _generate_hitting_time_queries(args, model, dataloader, file_suffix, guarantee_mark=False, save_seqs=False, num_item_to_query=1):
    all_queries, num_queries = {}, min(args.num_queries, len(dataloader))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        times, marks, next_item, mark_obs, last_time, next_time, max_T, up_to = \
            _setup_hitting_time_query(args, batch, model, use_tqdm=False, num_item_to_query=num_item_to_query)
        all_queries[i] = {
            'times': times, 'marks': marks, 'next_item': next_item, 'mark_obs': mark_obs,
            'last_time': last_time, 'next_time': next_time, 'max_T': max_T, 'up_to': up_to
        }
    save_results(args, all_queries, suffix=f'{file_suffix}_hitting_queries', save_seqs=save_seqs)
    return all_queries



def _hitting_time_eff_gt(args, model, queries):
    num_seqs = args.gt_num_seqs
    num_int_pts = args.gt_num_int_pts
    gts = []
    effs = []
    for i in tqdm(range(len(queries))):
        times, marks, next_item, mark_obs, last_time, next_time, max_T, up_to = queries[i].values()
        tmq = UnbiasedHittingTimeQuery(up_to=up_to, hitting_marks=next_item, batch_size=args.query_batch_size,
                                       device=args.device, use_tqdm=False, proposal_batch_size=args.proposal_batch_size)
        is_res = tmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks,
                              calculate_bounds=False)
        gts.append(is_res["est"].item())
        effs.append(is_res["rel_eff"].item())
    return gts, effs



def _hitting_time_ll_eff_pass(args, model, queries, num_seqs, num_int_pts):
    results = {"is_est": [],
               "is_var": [],
               "naive_est": [],
               "naive_var": [],
               "rel_eff": [],
               "avg_is_time": 0.0,
               "avg_naive_time": 0.0,
               "ll": [],
               "mark_obs": [],
               "ll_avg_time": 0.0}
    num_queries = len(queries)

    for i in tqdm(range(num_queries)):
        times, marks, next_item, mark_obs, last_time, next_time, max_T, up_to = queries[i].values()
        results['mark_obs'].append(mark_obs)

        # evaluating efficiency of hitting time queries up to whole obs. window
        tmq = UnbiasedHittingTimeQuery(up_to=up_to, hitting_marks=next_item, batch_size=args.query_batch_size,
                                       device=args.device, use_tqdm=False, proposal_batch_size=args.proposal_batch_size)
        is_t0 = time.perf_counter()
        is_cdf = tmq.estimate(model, num_seqs, num_int_pts, conditional_times=times, conditional_marks=marks,
                              calculate_bounds=False)
        is_t1 = time.perf_counter()
        is_time = (is_t1 - is_t0) / num_queries

        if not args.skip_naive:
            naive_t0 = time.perf_counter()
            naive_est = tmq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
            naive_t1 = time.perf_counter()

        results['is_est'].append(is_cdf['est'].item())
        results['is_var'].append(is_cdf['is_var'].item())
        results["naive_var"].append(is_cdf["naive_var"].item())
        results["rel_eff"].append(is_cdf["rel_eff"].item())
        results["avg_is_time"] += is_time
        if not args.skip_naive:
            results["naive_est"].append(naive_est)
            results["avg_naive_time"] += (naive_t1 - naive_t0) / num_queries
    return results



def hitting_time_queries_pass(args, model, dataloader, results):
    file_suffix = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    set_random_seed(seed=seed)

    print_log("Generating queries...")
    all_queries = _generate_hitting_time_queries(args, model, dataloader, file_suffix)

    if results is None:
        results = {'gt': None, 'gt_eff': None, 'estimates': {}}

    if not args.skip_gt:
        if (results['gt'] is None) or (results['gt_eff'] is None):
            print_log("Calculating gt...")
            results['gt'], results['gt_eff'] = _hitting_time_eff_gt(args, model, all_queries)
            save_results(args, results, file_suffix)
    else:
        print_log("Skipping GT Estimates.")

    print_log("Calculating est...")
    if not args.just_gt:
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = f'num_seqs_{num_seqs}'
            results['estimates'][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = f'num_int_pts_{num_int_pts}'
                args.seed = seed + i * len(args.num_int_pts) + j
                set_random_seed(args)
                if (ns_key in results['estimates']) and (np_key in results['estimates'][ns_key]) and (
                        results['estimates'][ns_key][np_key] is not None):
                    print_log(f'Skipping {ns_key} {np_key}')
                    continue
                else:
                    print_log(f'Estimating hitting queries for num_seqs={ns_key} and num_int_pts={np_key}')
                    results['estimates'][ns_key][np_key] = _hitting_time_ll_eff_pass(args, model, all_queries, num_seqs, num_int_pts)
                    save_results(args, results, file_suffix)
    save_results(args, results, file_suffix)
    return results



def hitting_time_queries_runtime(args, model, dataloader):
    file_suffix = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    set_random_seed(seed=seed)

    results = {}
    for hitting_query_item_pct in args.hitting_query_item_pcts:
        num_items_to_query = max(math.floor(model.num_channels * hitting_query_item_pct), 1)
        print_log(f"Generating queries for {num_items_to_query} items for hitting time queries...")
        all_queries = _generate_hitting_time_queries(args, model, dataloader, file_suffix, num_item_to_query=num_items_to_query)
        hit_key = f'pcts_items_{hitting_query_item_pct}'
        results[hit_key] = {}

        print_log("Calculating est...")
        for i, num_seqs in enumerate(args.num_seqs):
            ns_key = f'num_seqs_{num_seqs}'
            results[hit_key][ns_key] = {}
            for j, num_int_pts in enumerate(args.num_int_pts):
                np_key = f'num_int_pts_{num_int_pts}'
                args.seed = seed + i * len(args.num_int_pts) + j
                set_random_seed(args)
                print_log(f'Estimating hitting queries for num_seqs={ns_key} and num_int_pts={np_key}')
                results[hit_key][ns_key][np_key] = _hitting_time_ll_eff_pass(args, model, all_queries, num_seqs, num_int_pts)
                save_results(args, results, file_suffix)
    save_results(args, results, file_suffix)
    return results



def _setup_a_before_b_query(args, batch, model, use_tqdm=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    times, marks = batch["target_times"], batch["target_marks"]
    length = times.numel()
    to_condition_on = min(length - 1, 5)
    last_time = times[:, to_condition_on - 1].item()

    # pick A and B here
    condition_time, condition_marks = times[:, :to_condition_on], marks[:, :to_condition_on, ...]

    if args.checkpoint_path.startswith("../checkpoints/instacart_dept_norm"):
        perm = torch.randperm(model.num_channels)
        all_items = list(range(model.num_channels))
        A, B = all_items[perm[0]], all_items[perm[1]]
        max_T = times[:, -1].item()
        up_to = max_T - last_time
    else:
        existing_marks = torch.nonzero(condition_marks.sum(dim=-2), as_tuple=True)[1]
        if len(existing_marks) < 2:
            existing_marks = list(range(model.num_channels))

        perm = torch.randperm(len(existing_marks))
        A, B = existing_marks[perm[0]], existing_marks[perm[1]]

        up_to = max(min((times[:, to_condition_on].item() - last_time) * 10, 10.0), 1e-2)
        max_T = last_time + up_to


    remaining_times, remaining_marks = times[:, to_condition_on:], marks[:, to_condition_on:, ...]
    t_mask = remaining_times <= max_T
    remaining_times, remaining_marks = remaining_times[t_mask].unsqueeze(0), remaining_marks[t_mask].unsqueeze(0)

    # decide if A or B first
    if not remaining_times.shape[-1]:
        true_mark = [0, 0, 0, 1]
    else:
        A_idx = torch.nonzero(remaining_marks[:,:,A], as_tuple=True)[1]
        B_idx = torch.nonzero(remaining_marks[:,:,B], as_tuple=True)[1]
        if not len(A_idx) and not len(B_idx):
            true_mark = [0, 0, 0, 1]  # no a or b
        elif not len(A_idx):
            true_mark = [0, 0, 1, 0]  # b before a
        elif not len(B_idx):
            true_mark = [0, 1, 0, 0]  # a before b
        elif A_idx[0] == B_idx[0]:
            true_mark = [1, 0, 0, 0]  # a equals b
        elif A_idx[0] < B_idx[0]:
            true_mark = [0, 1, 0, 0]
        else:
            true_mark = [0, 0, 1, 0]

    return condition_time, condition_marks, A, B, up_to, true_mark  # (4)


def _generate_a_before_b_queries(args, model, dataloader, file_suffix):
    all_queries, num_queries = {}, min(args.num_queries, len(dataloader))
    dl_iter = iter(dataloader)
    for i in tqdm(range(num_queries)):
        batch = next(dl_iter)
        times, marks, A, B, up_to, true_mark = _setup_a_before_b_query(args, batch, model, use_tqdm=False)
        all_queries[i] = {'times': times, 'marks': marks, 'A': A, 'B': B, 'up_to': up_to, 'true_mark': true_mark}
    save_results(args, all_queries, suffix=f'{file_suffix}_ab_queries', save_seqs=False)
    return all_queries



def _a_before_b_queries_pass(args, model, queries, num_seqs, num_int_pts):
    results = {
        'is_est': [],
        "naive_est": [],
        'true_mark': [],
        'is_var': [],
        'naive_var': [],
        'rel_eff': [],
        'avg_is_time': 0.,
        'avg_naive_time': 0.

    }
    num_queries = len(queries)

    for i in tqdm(range(num_queries)):
        times, marks, A, B, up_to, true_mark = queries[i].values()
        results['true_mark'].append(true_mark)

        abq = AbeforeBQuery(up_to, torch.tensor([A]), torch.tensor([B]), batch_size=args.query_batch_size, device=args.device, use_tqdm=False, proposal_batch_size=args.proposal_batch_size)
        is_t0 = time.perf_counter()
        is_est = abq.estimate(model, num_seqs, num_int_pts, times, marks)
        is_t1 = time.perf_counter()

        if not args.skip_naive:
            naive_t0 = time.perf_counter()
            naive_est = abq.naive_estimate(model, num_seqs, conditional_times=times, conditional_marks=marks)
            naive_t1 = time.perf_counter()
            results['naive_est'].append([naive_est['a_equals_b'], naive_est['a_before_b'], naive_est['b_before_a'], naive_est['no_a_or_b']])
            results['avg_naive_time'] += (naive_t1 - naive_t0) / num_queries

        results['is_est'].append([is_est['a_equals_b'], is_est['a_before_b'], is_est['b_before_a'], is_est['no_a_or_b']])
        results['is_var'].append(is_est['is_var'])
        results['naive_var'].append(is_est['naive_var'])
        # results['rel_eff'].append(is_est['rel_eff'])
        results["avg_is_time"] += (is_t1 - is_t0) / num_queries
    return results



def a_before_b_queries_pass(args, model, dataloader, partial_res):
    file_suffix = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    seed = args.seed
    set_random_seed(seed=seed)

    print_log("Generating queries...")
    all_queries = _generate_a_before_b_queries(args, model, dataloader, file_suffix)
    results = {}

    print_log("Calculating est...")
    for i, num_seqs in enumerate(args.num_seqs):
        ns_key = f'num_seqs_{num_seqs}'
        results[ns_key] = {}
        for j, num_int_pts in enumerate(args.num_int_pts):
            np_key = f'num_int_pts_{num_int_pts}'
            args.seed = seed + i * len(args.num_int_pts) + j
            set_random_seed(args)
            if (ns_key in results) and (np_key in results[ns_key]) and (results[ns_key][np_key] is not None):
                print_log(f'Skipping {ns_key} {np_key}')
                continue
            else:
                print_log(f'Estimating A before B queries for num_seqs={ns_key} and num_int_pts={np_key}')
                results[ns_key][np_key] = _a_before_b_queries_pass(args, model, all_queries, num_seqs, num_int_pts)
                save_results(args, results, file_suffix)
    save_results(args, results, file_suffix)
    return results



def main():
    print_log("Getting arguments.")
    args = get_args()
    args = get_training_args(args)

    args.evaluate = True
    args.top_k = 0
    args.top_p = 0
    args.batch_size = 1
    args.shuffle = True  # shuffle the test dataloader

    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    args.pin_test_memory = True
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)

    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, _, _ = setup_model_and_optim(args, len(train_dataloader))

    load_checkpoint(args, model)
    model.eval()

    if args.continue_experiments is not None:
        partial_res = pickle.load(open(args.continue_experiments, "rb"))
    else:
        partial_res = None

    print_log("")
    print_log("")
    print_log("Commencing Experiments")
    with torch.no_grad():
        if args.hitting_time_queries:
            # results = hitting_time_queries_pass(args, model, valid_dataloader, partial_res)
            results = hitting_time_queries_pass(args, model, test_dataloader, partial_res)
            print("Hitting time queries done.")
            # print(results)
        elif args.a_before_b_queries:
            # results = a_before_b_queries_pass(args, model, valid_dataloader, partial_res)
            results = a_before_b_queries_pass(args, model, test_dataloader, partial_res)
            print("A before B queries done.")
        elif args.runtime_hitting_time_queries:
            # results = hitting_time_queries_runtime(args, model, valid_dataloader)
            results = hitting_time_queries_runtime(args, model, test_dataloader)
            print("Hitting time runtime experiments done.")

if __name__ == "__main__":
    main()

