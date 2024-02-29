import os
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad
from tqdm import tqdm
import pickle
import datetime
from lib.data import PointPatternDataset, pad_and_combine_instances
from lib.arguments import get_args
from lib.utils import *


def forward_pass(args, batch, model, sample_timestamps=None, num_samples=150, get_raw_likelihoods=False):
    if args.cuda:
        batch = {k: v.cuda(torch.cuda.current_device()) for k, v in batch.items()}

    padding_mask = batch["padding_mask"]
    tgt_marks, tgt_timestamps = batch["target_marks"], batch["target_times"]
    pp_id = batch["pp_id"]

    T = batch["T"]

    if sample_timestamps is None:
        sample_timestamps = torch.rand(
            tgt_timestamps.shape[0],
            num_samples,
            dtype=tgt_timestamps.dtype,
            device=tgt_timestamps.device
        ).clamp(min=1e-8) * T  # ~ U(0, T)

    # Forward Pass
    results = model(
        marks=tgt_marks,
        timestamps=tgt_timestamps,
        sample_timestamps=sample_timestamps,
    )

    # Calculate losses
    ll_results = model.log_likelihood(
        return_dict=results,
        target_marks = tgt_marks,
        right_window=T,
        left_window=0.0,
        mask=padding_mask,
        # reduce=not get_raw_likelihoods,
        normalize_by_window=args.normalize_by_window,
        normalize_by_events=args.normalize_by_events,
        gamma=args.gamma
    )

    if get_raw_likelihoods:
        return ll_results, sample_timestamps, tgt_timestamps

    log_likelihood, ll_mark_contrib, ll_time_contrib, ll_time_pos, ll_time_neg = \
        ll_results["log_likelihood"], ll_results["positive_contribution_marks"], ll_results["timing_contribution"], \
        ll_results["positive_contribution_pp"], ll_results["negative_contribution"]

    if args.only_train_on_time:
        loss = -1 * ll_time_contrib
    else:
        loss = -1 * log_likelihood  # minimize loss, maximize log likelihood

    return_dict = {
        "loss": loss,
        "log_likelihood": log_likelihood,
        "ll_mark": ll_mark_contrib,
        "ll_time": ll_time_contrib,
        "ll_time_pos": ll_time_pos,
        "ll_time_neg": ll_time_neg
    }

    if args.normalize_by_events:
        return_dict["ll_mark_norm"] = ll_results["positive_contribution_marks_norm"]

    return return_dict, results


def backward_pass(args, loss, model, optimizer):
    optimizer.zero_grad()

    if torch.isnan(loss).any().item():
        return False
    else:
        loss.backward()
        clip_grad(parameters=model.parameters(), max_norm=args.grad_clip, norm_type=2)
        return True


def analyze_gradient_scale(loss, model, optimizer, epoch_number):
    print(f'Start printing gradient scales for epoch: {epoch_number}')
    for loss_name, loss_component in loss.items():
        optimizer.zero_grad()
        loss_component.backward(retain_graph=True)
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if name.startswith('hidden_to_item_logits') or name.startswith('decoder.channel_embedding'):
                continue
            else:
                # Consider having check to see if `p` comes from the main model or the mark-specific component
                # If `p` is only used for the mark-distribution, then don't add it to the total_grad_norm
                param_norm = param.grad.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1./2)
        print_log(f'The scale of {loss_name} is: {total_grad_norm}')

        if loss_name == 'll_mark':
            mark_grad = total_grad_norm
        elif loss_name == 'll_time':
            time_grad = total_grad_norm

    optimizer.zero_grad()
    print("Finish printing gradients")
    return (mark_grad/time_grad)



def train_step(args, model, optimizer, lr_scheduler, batch, epoch_number, print_gradient, gradients_ratio_mark_to_time):
    loss_results, forward_results = forward_pass(args, batch, model)
    if args.analyze_gradient and print_gradient:  # check the scale of gradient every epoch
        gradients_ratio = analyze_gradient_scale(loss_results, model, optimizer, epoch_number)
        gradients_ratio_mark_to_time.append(gradients_ratio)
        save_results(args, gradients_ratio_mark_to_time, save_gradients=True)

    if backward_pass(args, loss_results["loss"], model, optimizer):
        optimizer.step()
        lr_scheduler.step()
    else:
        print_log('======= NAN-Loss =======')
        print_log("Loss Results:",
                  {k: (torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k, v in loss_results.items() if
                   isinstance(v, torch.Tensor)})
        print_log("Loss Results:", loss_results)
        print_log("")
        print_log("Batch:",
                  {k: (torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k, v in batch.items() if
                   isinstance(v, torch.Tensor)})
        print_log("Batch:", batch)
        print_log("")
        print_log("Results:", {k: (torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k, v in
                               forward_results["state_dict"].items()})
        print_log("Results:", {k: (torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k, v in
                               forward_results["intensities"].items()})
        print_log("Results:", {k: (torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k, v in
                               forward_results["sample_intensities"].items()})
        print_log("Results:", forward_results)
        print_log("")
        print_log("========================")
        print_log("")
        print_log("Set embeddings: ")
        if torch.isnan(model.channel_embedding.weight.data).any():
            print('Detect NaN values in embedding parameters.')
            print(model.channel_embedding.weight.data)
        if torch.isinf(model.channel_embedding.weight.data).any():
            print('Detect inf values in embedding parameters.')
            print(model.channel_embedding.weight.data)
        print_log("")
        print_log("Model parameters: ")
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f'Detect NaN values in {name} parameters.')
                print(param.data)
            if torch.isinf(param.data).any():
                print(f'Detect inf values in {name} parameters.')
                print(param.data)
        print_log("========================")
        input()

    return loss_results, gradients_ratio_mark_to_time



def train_epoch(args, model, optimizer, lr_scheduler, dataloader, epoch_number, gradients_ratio_mark_to_time, print_gradient=True):
    model.train()
    total_losses = defaultdict(lambda: 0.0)
    data_len = len(dataloader)
    for i, batch in enumerate(dataloader):
        batch_loss, gradients_ratio_mark_to_time \
            = train_step(args, model, optimizer, lr_scheduler, batch, epoch_number, print_gradient, gradients_ratio_mark_to_time)
        print_gradient = False
        for k, v in batch_loss.items():
            total_losses[k] += v.item()
        if (((i + 1) % args.log_interval == 0) or ((i + 1 <= 5) and (epoch_number <= 1)) or
            ((i + 1) % args.log_interval != 0 and (i + 1) == data_len)):
            items_to_print = [("LR", lr_scheduler.get_lr())]
            items_to_print.extend([(k, v / (i+1)) for k, v in total_losses.items()])
            print_results(args, items_to_print, epoch_number, i + 1, data_len, True)
    return {k: v / data_len for k, v in total_losses.items()}



def eval_epoch(args, model, eval_dataloader, epoch_number, num_samples=150):
    model.eval()

    with torch.no_grad():
        total_losses = defaultdict(lambda: 0.0)
        data_len = len(eval_dataloader)
        for i, batch in enumerate(eval_dataloader):
            batch_loss, results = forward_pass(args, batch, model, sample_timestamps=None, num_samples=num_samples)

            for k, v in batch_loss.items():
                total_losses[k] += v.item()

    print_results(args, [(k, v / data_len) for k, v in total_losses.items()], epoch_number, i + 1, data_len, False)
    return {k: v / data_len for k, v in total_losses.items()}



if __name__ == "__main__":
    file_suffix = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

    print_log("Getting arguments.")
    args = get_args(file_suffix)

    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)

    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, optimizer, lr_scheduler = setup_model_and_optim(args, len(train_dataloader))

    if not args.poisson:
        report_model_stats(model)

        if args.finetune:
            epoch = load_checkpoint(args, model)
        else:
            epoch = 0
        original_epoch = epoch

        print_log("Starting training.")
        results = {"valid": [], "train": [], "test": []}
        last_valid_ll = -float('inf')
        gradients_ratio_mark_to_time = []
        epsilon = 0.03

        pbar = tqdm(total=args.train_epochs)
        while epoch < args.train_epochs or args.early_stop:
            results["train"].append(train_epoch(args, model, optimizer, lr_scheduler, train_dataloader, epoch + 1, gradients_ratio_mark_to_time))

            if args.do_valid and ((epoch + 1) % args.valid_epochs == 0):
                new_valid = eval_epoch(args, model, valid_dataloader, epoch + 1)
                results["valid"].append(new_valid)
                if args.early_stop:
                    if new_valid["log_likelihood"] - last_valid_ll < epsilon:
                        break
                last_valid_ll = new_valid["log_likelihood"]

            if ((epoch + 1) % args.save_epochs == 0):
                save_checkpoint(args, model, optimizer, lr_scheduler, epoch + 1)

            if (epoch + 1) % args.save_epochs == 0:
                with open(f"{args.checkpoint_path}/{args.set_assumption}_train_results.pickle", 'wb') as f:
                    pickle.dump(results, f)

            epoch += 1
            pbar.update(1)
        pbar.close()

        if args.save_epochs > 0 and original_epoch != epoch and epoch % args.save_epochs != 0:
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch)
    else:
        model.get_poisson_statistics(train_dataloader)
        epoch = 0
        save_checkpoint(args, model, optimizer, lr_scheduler, epoch)
        results = {"valid": [], "train": [], "test": []}
        results["train"].append(eval_epoch(args, model, train_dataloader, epoch + 1, num_samples=500))
        results["valid"].append(eval_epoch(args, model, valid_dataloader, epoch + 1, num_samples=500))


    if args.do_valid:
        reps = 5
        for _ in range(reps):
            test_results = eval_epoch(args, model, test_dataloader, epoch + 1, num_samples=500)
            results["test"].append(test_results)

    del model
    del optimizer
    del lr_scheduler
    del train_dataloader
    del valid_dataloader
    del test_dataloader
    torch.cuda.empty_cache()


    with open(f"{args.checkpoint_path}/{args.set_assumption}_train_results.pickle", 'wb') as f:
        pickle.dump(results, f)

    # print(results)
