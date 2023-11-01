import csv
import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Literal

rel_map_14 = defaultdict(lambda: -1, {
        "Comparison.Concession": 0,
        "Comparison.Contrast": 1,

        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 3,
        "Contingency.Condition": 4,

        "Expansion.Alternative": 5,
        "Expansion.Alternative.Chosen alternative": 6,
        "Expansion.Conjunction": 7,
        "Expansion.Exception": 8,
        "Expansion.Instantiation": 9,
        "Expansion.Restatement": 10,

        # "Temporal",
        "Temporal.Asynchronous.Precedence": 11,
        "Temporal.Asynchronous.Succession": 12,
        "Temporal.Synchrony": 13})

    
def preprocess_conll(json_file_path: str, out_file_path: str, 
                    relation_types: List[Literal['AltLex', 'EntRel', 'Explicit', 'Implicit', 'NoRel']],
                    allow_multi_rel: bool = True):

    rel_map = rel_map_14

    df = pd.read_json(json_file_path, lines=True)

    if relation_types:
        df = df[df["Type"].isin(set(relation_types))]
    df.fillna("", inplace=True)
    
    result_dicts = []
    for idx, row in tqdm(df.iterrows()):
        rel_strs = row[5]
        
        arg1 = row[0]["RawText"]
        arg2 = row[1]["RawText"]
        class_set = set()
        
        for class_str in rel_strs:
            class_id = rel_map[class_str]
            if class_id == -1:
                continue
            class_set.add(class_id)
            
        class_set = list(class_set)
        class_set=list(map(lambda x:str(x),class_set))
         
        if len(class_set) == 0:
            continue
        
        if allow_multi_rel:
                for c in class_set:
                    d = {'arg1': arg1, 'arg2': arg2, 'label': c}
                    result_dicts.append(d)
        else:
            d = {'arg1': arg1, 'arg2': arg2, 'label': class_set[0]}
            result_dicts.append(d)
                       
    with open(out_file_path, 'w', encoding='utf-8', newline='') as out_file:
        fieldnames = ['arg1', 'arg2', 'label']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames,delimiter='\t')

        # writer.writeheader()
        writer.writerows(result_dicts)


def preprocess_conll_for_test(json_file_path: str, out_file_path: str,
                             relation_types: List[Literal['AltLex', 'EntRel', 'Explicit', 'Implicit', 'NoRel']],
                             allow_multi_rel: bool = True):
    
    rel_map = rel_map_14

    df = pd.read_json(json_file_path, lines=True)

    if relation_types:
        df = df[df["Type"].isin(set(relation_types))]
    df.fillna("", inplace=True)
    
    result_dicts = []
    for idx, row in tqdm(df.iterrows()):
        rel_strs = row[5]
        arg1 = row[0]["RawText"]
        arg2 = row[1]["RawText"]
        class_set = set()
        
        for class_str in rel_strs:
            class_id = rel_map[class_str]
            if class_id == -1:
                continue
            class_set.add(class_id)
            
        class_set = list(class_set)
        class_set=list(map(lambda x:str(x),class_set))
         
        if len(class_set) == 0:
            continue
        
        d = {'arg1': arg1, 'arg2': arg2, 'label': '#'.join(class_set)}
        result_dicts.append(d)
                       
    with open(out_file_path, 'w', encoding='utf-8', newline='') as out_file:
        fieldnames = ['arg1', 'arg2', 'label']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames,delimiter='\t')

        # writer.writeheader()
        writer.writerows(result_dicts)


def preprocess_data(args):
    logging.info('preprocessing data...')

    
    if not os.path.exists(args.dataset_dir_path):
        os.makedirs(args.dataset_dir_path)
    
    preprocess_conll(args.original_train_dataset_path, os.path.join(args.dataset_dir_path, "train.tsv"),
                             relation_types=['Implicit'],
                             allow_multi_rel=True
                             )
    
    preprocess_conll_for_test(args.original_dev_dataset_path, os.path.join(args.dataset_dir_path, "dev.tsv"),
                             relation_types=['Implicit'],
                             allow_multi_rel=False
                             )
    
    preprocess_conll_for_test(args.original_test_dataset_path, os.path.join(args.dataset_dir_path, "test.tsv"),
                             relation_types=['Implicit'],
                             allow_multi_rel=False,
                             )
    preprocess_conll_for_test(args.original_blind_dataset_path, os.path.join(args.dataset_dir_path, "blind.tsv"),
                             relation_types=['Implicit'],
                             allow_multi_rel=False,
                             )
   
    logging.info('done')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_train_dataset_path", type=str,
                        help="Original train dataset path")
    parser.add_argument("--original_dev_dataset_path", type=str,
                        help="Original dev dataset path")
    parser.add_argument("--original_test_dataset_path", type=str,
                        help="Original test dataset path")
    parser.add_argument("--original_blind_dataset_path", type=str,
                        help="Original blind dataset path")
    parser.add_argument("--dataset_dir_path", type=str,
                        help="Dataset directory path")
    parser.add_argument("--num_rels", type=int, default=14,
                        choices=[14],
                        help="how many relations are computed")
    args = parser.parse_args()
    preprocess_data(args)
