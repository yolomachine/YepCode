#!/usr/bin/python

import os
import random
import argparse
import subprocess


def chunks(lst, n):
    n = (len(lst) + n - 1) // n

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


parser = argparse.ArgumentParser()
parser.add_argument('num_processes', type=int, help='Number of processes')
parser.add_argument('dataset_path', help='Path to dataset')
parser.add_argument('list_file', type=str, help='Path to list file')
parser.add_argument('output_folder', type=str, help='Path to output folder')
parser.add_argument('--skip_tokens', help='Tokens to replace skip', nargs='*', default=set())
args = parser.parse_args()

data = [i.strip() for i in open(args.list_file).readlines()]
random.shuffle(data)

for i, lst in enumerate(chunks(data, args.num_processes)):
    path = os.path.join(args.output_folder, str(i))
    os.makedirs(path, exist_ok=True)

    list_file = os.path.join(path, 'list.txt')
    with open(list_file, 'w') as file:
        print(*lst, sep='\n', file=file)

    subprocess.Popen(['python', 'preprocessor.py', args.dataset_path, list_file, os.path.join(path, 'tokenized.txt'), '--skip_tokens', *args.skip_tokens])
