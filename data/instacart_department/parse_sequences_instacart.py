import argparse
import csv
import random
import numpy
import datetime
import json
import zipfile
from io import TextIOWrapper
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
import pickle
import re

random.seed(0)
numpy.random.seed(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat_dir', default='./', type=str,
                        help="Directory where the data files are located.")
    parser.add_argument('--out_dir', default='./', type=str, help="Directory where results will be saved.")
    parser.add_argument('--min_events', default=5, type=int,
                        help="Minimum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--max_events', default=200, type=int,
                        help="Maximum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--time_norm', default=1, type=int,
                        help="Default is for 1 unit == 1 day.")
    parser.add_argument('--valid_pct', default=0.1, type=float,
                        help="Percentage of sequences to set aside for validation split.")
    parser.add_argument('--test_pct', default=0.15, type=float,
                        help="Percentage of sequences to set aside for test split.")
    args = parser.parse_args()
    args.dat_dir = args.dat_dir.rstrip("/")
    args.out_dir = args.out_dir.rstrip("/")
    return args


def collect_sequences(args):
    sequences = defaultdict(list)
    first_line = True

    # first event naturally starts from some t > 0
    pbar = tqdm(total=3325578)  # 3421083 rows in orders
    with open(f'{args.dat_dir}/df_orders_time_and_sets_final.csv', "r", encoding='UTF-8') as f:
        for event in tqdm(f):
            if first_line:
                first_line = False
                continue
            user_id, timestamp, dept_list = event.split(',', 2)
            item_set = ''.join(c for c in dept_list if c not in '"\n').strip('][').split(', ')
            item_set = [int(x) - 1 for x in item_set]  # map so that it starts from 0
            user_id, timestamp = int(user_id), float(timestamp)

            sequences[user_id].append([timestamp, item_set])
            pbar.update(1)
    pbar.close()
    return sequences


def process_sequences(args, sequences):
    print("   Num Sequences prior to Filtering:", len(sequences))
    for u in tqdm(list(sequences.keys())):
        if args.min_events <= len(sequences[u]) <= args.max_events:
            sequences[u] = list(zip(*sorted(sequences[u], key=lambda x: x[
                0])))  # Sort by timestamps and then split into two lists: one for timestmaps and one for marks

            sequences[u] = (
                [(t - sequences[u][0][0]) / args.time_norm for t in sequences[u][0][1:]],
                sequences[u][1][1:],
            )
        else:
            del sequences[u]
    print("   Num Sequences after Filtering:", len(sequences))
    return sequences


def format_sequence(args, sequence, user_id):
    return json.dumps({
        "user": user_id,
        "T": sequence[0][-1],
        "times": sequence[0],
        "marks": sequence[1],
    }) + "\n"


if __name__ == "__main__":
    args = get_args()

    print("Gathering Event Sequences")
    sequences = collect_sequences(args)

    print("Processing Event Sequences")
    sequences = process_sequences(args, sequences)

    print("Saving Results")
    all_strs, train_strs, valid_strs, test_strs = [], [], [], []

    sequences = list(sequences.items())
    random.shuffle(sequences)

    for i, (user_id, sequence) in tqdm(enumerate(sequences)):
        seq_str = format_sequence(args, sequence, user_id)
        all_strs.append(seq_str)
        progress = (i + 1) / len(sequences)
        if progress <= args.test_pct:
            test_strs.append(seq_str)
        elif progress <= args.test_pct + args.valid_pct:
            valid_strs.append(seq_str)
        else:
            train_strs.append(seq_str)

    dir_strs = [
        ("{}/all_sequences.jsonl", all_strs),
        ("{}/train/train_instacart.jsonl", train_strs),
        ("{}/valid/valid_instacart.jsonl", valid_strs),
        ("{}/test/test_instacart.jsonl", test_strs),
    ]

    for dir, strs in dir_strs:
        with open(dir.format(args.out_dir), "w") as f:
            f.writelines(strs)
