#!/usr/bin/python

import os
import json
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to dataset')
parser.add_argument('output', type=str, help='Path to output file')
args = parser.parse_args()

list_file = open(args.output, 'w')

for i in tqdm.tqdm(os.listdir(args.path)):
    subfolder = os.path.join(args.path, i)
    if not os.path.exists(os.path.join(subfolder, 'cpp')):
        continue
    
    meta = json.load(open(os.path.join(subfolder, 'meta.json')))
    for value in tqdm.tqdm(meta.get('Submissions', {}).values(), leave=False):
        if value['Download-Status'] != 'FINISHED':
            continue
            
        print(i, value['Id'], *(i.replace(' ', '_') for i in value['Tags']), file=list_file)
