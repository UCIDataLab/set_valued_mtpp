import os
import json
import pickle
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from lib.data import PointPatternDataset, pad_and_combine_instances
from lib.models import get_model
from lib.optim import get_optimizer, get_lr_scheduler

def set_random_seed(args=None, seed=None):
    """Set random seed for reproducibility."""

    if (seed is None) and (args is not None):
        seed = args.seed

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def print_log(*args):
    print("[{}]".format(datetime.now()), *args)


def print_results(args, items, epoch_number, iteration, data_len, training=True):
    msg = "[{}] Epoch {}/{} | Iter {}/{} | ".format("T" if training else "V", epoch_number, args.train_epochs, iteration, data_len)
    msg += "".join("{} {:.4E} | ".format(k, v) for k,v in items)
    print_log(msg)



def get_data(args):
    train_dataset = PointPatternDataset(
        file_path=args.train_data_path,
        args=args,
        keep_pct=args.train_data_percentage,
        set_dominating_rate=args.evaluate,
        is_test=False,
    )
    args.num_channels = train_dataset.vocab_size
    if args.force_num_channels is not None:
        args.num_channels = args.force_num_channels

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda x: pad_and_combine_instances(x, args.num_channels),
        drop_last=True,
    )

    args.max_period = train_dataset.get_max_T() / 2.0

    print_log("Loaded {} / {} training examples / batches from {}".format(len(train_dataset), len(train_dataloader), args.train_data_path))

    if args.do_valid:
        valid_dataset = PointPatternDataset(
            file_path=args.valid_data_path,
            args=args,
            keep_pct=1.0,
            set_dominating_rate=False,
            is_test=False,
        )

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda x: pad_and_combine_instances(x, args.num_channels),
            drop_last=True,
        )
        print_log("Loaded {} / {} validation examples / batches from {}".format(len(valid_dataset), len(valid_dataloader), args.valid_data_path))

        test_dataset = PointPatternDataset(
            file_path=args.test_data_path,
            args=args,
            keep_pct=1.0,  # object accounts for the test set having (1 - valid_to_test_pct) amount
            set_dominating_rate=False,
            is_test=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=max(args.batch_size // 4, 1),
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda x: pad_and_combine_instances(x, args.num_channels),
            drop_last=True,
            pin_memory=args.pin_test_memory,
        )
        # print_log("Loaded {} / {} test examples / batches from {}".format(len(test_dataset), len(test_dataloader), args.valid_data_path))
        print_log("Loaded {} / {} test examples / batches from {}".format(len(test_dataset), len(test_dataloader), args.test_data_path))
    else:
        valid_dataloader = None
        test_dataloader = None


    return train_dataloader, valid_dataloader, test_dataloader


def save_results(args, results, suffix="", save_args=False, save_models=False, save_seqs=False, save_gradients=False):
    if save_args:  # save experiment settings
        os.mkdir(args.checkpoint_path)
        fp = f"{args.checkpoint_path.rstrip('/')}/arguments.txt"
        with open(fp, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print_log("Saved results at {}".format(fp))
    elif save_gradients:
        fp = f"{args.checkpoint_path.rstrip('/')}/gradients_ratio_mark_to_time.pickle"
        with open(fp, "wb") as f:
            pickle.dump(results, f)
        print(f"Gradients saved at {fp}")
    elif save_seqs:
        with open(f"{args.checkpoint_path.rstrip('/')}/simulated_seqs{'_' + suffix if suffix != '' else ''}.pickle", 'wb') as f:
            pickle.dump(results, f)
    else:
        fp = f"{args.checkpoint_path.rstrip('/')}/results"
        if args.next_set_prediction:
            fp += "/next_set_prediction"
        elif args.hitting_time_queries:
            fp += "/hitting_time"
        elif args.a_before_b_queries:
            fp += "/a_before_b"
        elif args.runtime_hitting_time_queries:
            fp += "/hitting_time_runtime"

        folders = fp.split("/")
        for i in range(len(folders)):
            if folders[i] == "":
                continue
            intermediate_path = "/".join(folders[:i + 1])
            if not os.path.exists(intermediate_path):
                os.mkdir(intermediate_path)

        fp += "/results{}{}.pickle".format("_" if suffix != "" else "", suffix)
        with open(fp, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved at {fp}")


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    # Create folder if not already created
    folder_path = args.checkpoint_path
    folders = folder_path.split("/")
    for i in range(len(folders)):
        if folders[i] == "":
            continue
        intermediate_path = "/".join(folders[:i+1])
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)

    final_path = "{}/model_{:03d}.pt".format(folder_path.rstrip("/"), epoch)
    if os.path.exists(final_path):
        os.remove(final_path)
    torch.save(model.state_dict(), final_path)
    print_log("Saved model at {}".format(final_path))


def load_checkpoint(args, model):
    folder_path = args.checkpoint_path
    if not os.path.exists(folder_path):
        print_log(f"Checkpoint path [{folder_path}] does not exist.")
        return 0

    print_log(f"Checkpoint path [{folder_path}] does exist.")
    files = [f for f in os.listdir(folder_path) if ".pt" in f]
    if len(files) == 0:
        print_log("No .pt files found in checkpoint path.")
        return 0

    latest_model = sorted(files)[-1]
    file_path = "{}/{}".format(folder_path.rstrip("/"), latest_model)

    if not os.path.exists(file_path):
        print_log(f"File [{file_path}] not found.")
        return 0

    model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
    if args.cuda:
        model.cuda(torch.cuda.current_device())
    print_log("Loaded model from {}".format(file_path))
    return int(latest_model.replace("model_", "").replace(".pt", "")) + 1


def setup_model_and_optim(args, epoch_len):
    model = get_model(
        channel_embedding_size=args.channel_embedding_size,
        num_channels=args.num_channels,
        dec_recurrent_hidden_size=args.dec_recurrent_hidden_size,
        set_assumption=args.set_assumption,
        condition_on_history=args.condition_on_history,
        neural_hawkes=args.neural_hawkes,
        rmtpp=args.rmtpp,
        poisson=args.poisson,
        dyn_dom_buffer=args.dyn_dom_buffer,
        num_layer_item=args.num_layer_item
    )

    if args.cuda:
        print(f"USING GPU {args.device_num}")
        torch.cuda.set_device(args.device_num)
        model.cuda(torch.cuda.current_device())

    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_len)

    return model, optimizer, lr_scheduler



def report_model_stats(model):
    encoder_parameter_count = 0
    aggregator_parameter_count = 0
    decoder_parameter_count = 0
    total = 0
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            encoder_parameter_count += param.numel()
        elif name.startswith("aggregator"):
            aggregator_parameter_count += param.numel()
        else:
            decoder_parameter_count += param.numel()
        total += param.numel()

    print_log()
    print_log("<Total Parameter Count>: {}".format(decoder_parameter_count))
    print_log()


def get_training_args(args):
    '''
    Get the arguments used to train the model depending on the checkpoint_path
    '''
    model_args_path = args.checkpoint_path.rstrip('/') + '/arguments.txt'
    training_args_dict = {'train_data_path', 'valid_data_path', 'test_data_path', 'channel_embedding_size',
                          'num_channels', 'dec_recurrent_hidden_size', 'set_assumption', 'condition_on_history',
                          'mixture_w_dim', 'num_layer_item'}
    with open(model_args_path) as f:
        model_args = json.load(f)
    for k, v in model_args.items():
        if k in training_args_dict:
            args.__dict__[k] = v
    return args