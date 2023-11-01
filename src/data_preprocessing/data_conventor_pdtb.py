import csv
import os
import json
import logging
import argparse
from collections import defaultdict
from typing import List, Literal

rel_map_14 = defaultdict(lambda: -1, {
    # "Comparison"
    "Comparison.Concession": 0,
    "Comparison.Concession.Contra-expectation": 0,
    "Comparison.Concession.Expectation": 0,
    "Comparison.Pragmatic concession": 0,
    "Comparison.Contrast": 1,
    "Comparison.Contrast.Juxtaposition": 1,
    "Comparison.Contrast.Opposition": 1,
    "Comparison.Pragmatic contrast": 1,

    # "Contingency"
    "Contingency.Cause.Reason": 2,
    "Contingency.Pragmatic cause.Justification": 2,
    "Contingency.Cause.Result": 3,
    "Contingency.Condition": 4,
    "Contingency.Condition.Hypothetical": 4,
    "Contingency.Pragmatic condition.Relevance": 4,
    # "Contingency.Cause",

    # "Expansion"
    "Expansion.Alternative": 5,
    "Expansion.Alternative.Conjunctive": 5,
    "Expansion.Alternative.Disjunctive": 5,
    "Expansion.Alternative.Chosen alternative": 6,
    "Expansion.Conjunction": 7,
    "Expansion.List": 7,
    "Expansion.Exception": 8,
    "Expansion.Instantiation": 9,
    "Expansion.Restatement": 10,
    "Expansion.Restatement.Equivalence": 10,
    "Expansion.Restatement.Generalization": 10,
    "Expansion.Restatement.Specification": 10,

    # "Temporal",
    "Temporal.Asynchronous.Precedence": 11,
    "Temporal.Asynchronous.Succession": 12,
    "Temporal.Synchrony": 13})


rel_map_11 = defaultdict(lambda: -1, {
    # "Comparison",
    "Comparison.Concession": 0,
    "Comparison.Concession.Contra-expectation": 0,
    "Comparison.Concession.Expectation": 0,
    "Comparison.Contrast": 1,
    "Comparison.Contrast.Juxtaposition": 1,
    "Comparison.Contrast.Opposition": 1,
    # "Comparison.Pragmatic concession",
    # "Comparison.Pragmatic contrast",

    # "Contingency",
    "Contingency.Cause": 2,
    "Contingency.Cause.Reason": 2,
    "Contingency.Cause.Result": 2,
    "Contingency.Pragmatic cause.Justification": 3,
    # "Contingency.Condition",
    # "Contingency.Condition.Hypothetical",
    # "Contingency.Pragmatic condition.Relevance",

    # "Expansion",
    "Expansion.Alternative": 4,
    "Expansion.Alternative.Chosen alternative": 4,
    "Expansion.Alternative.Conjunctive": 4,
    "Expansion.Conjunction": 5,
    "Expansion.Instantiation": 6,
    "Expansion.List": 7,
    "Expansion.Restatement": 8,
    "Expansion.Restatement.Equivalence": 8,
    "Expansion.Restatement.Generalization": 8,
    "Expansion.Restatement.Specification": 8,
    # "Expansion.Alternative.Disjunctive",
    # "Expansion.Exception",

    # "Temporal",
    "Temporal.Asynchronous.Precedence": 9,
    "Temporal.Asynchronous.Succession": 9,
    "Temporal.Synchrony": 10})

rel_map_4 = defaultdict(lambda: -1, {
    "Comparison": 0,
    "Comparison.Concession": 0,
    "Comparison.Concession.Contra-expectation": 0,
    "Comparison.Concession.Expectation": 0,
    "Comparison.Contrast": 0,
    "Comparison.Contrast.Juxtaposition": 0,
    "Comparison.Contrast.Opposition": 0,
    "Comparison.Pragmatic concession": 0,
    "Comparison.Pragmatic contrast": 0,

    "Contingency": 1,
    "Contingency.Cause": 1,
    "Contingency.Cause.Reason": 1,
    "Contingency.Cause.Result": 1,
    "Contingency.Condition": 1,
    "Contingency.Condition.Hypothetical": 1,
    "Contingency.Pragmatic cause.Justification": 1,
    "Contingency.Pragmatic condition.Relevance": 1,

    "Expansion": 2,
    "Expansion.Alternative": 2,
    "Expansion.Alternative.Chosen alternative": 2,
    "Expansion.Alternative.Conjunctive": 2,
    "Expansion.Alternative.Disjunctive": 2,
    "Expansion.Conjunction": 2,
    "Expansion.Exception": 2,
    "Expansion.Instantiation": 2,
    "Expansion.List": 2,
    "Expansion.Restatement": 2,
    "Expansion.Restatement.Equivalence": 2,
    "Expansion.Restatement.Generalization": 2,
    "Expansion.Restatement.Specification": 2,

    "Temporal": 3,
    "Temporal.Asynchronous.Precedence": 3,
    "Temporal.Asynchronous.Succession": 3,
    "Temporal.Synchrony": 3})
    
def preprocess_pdtb(csv_file_path: str, out_file_path: str, sections: List[int], num_labels: Literal[4, 11, 14],
                    relation_types: List[Literal['AltLex', 'EntRel', 'Explicit', 'Implicit', 'NoRel']],
                    allow_multi_rel: bool = True):
    if num_labels == 4:
        rel_map = rel_map_4
    elif num_labels == 11:
        rel_map = rel_map_11
    elif num_labels == 14:
        rel_map = rel_map_14
    else:
        raise "num_labels can only selected in [4,11,14]"
    with open(csv_file_path, 'r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        result_dicts = []
        for row in csv_reader:
            section = int(row['Section'])
            relation = row['Relation']
            if (section not in sections) or (relation not in relation_types):
                continue
            arg1 = row['Arg1_RawText']
            arg2 = row['Arg2_RawText']

            if relation == 'Implicit':
                conn1 = row['Conn1']
            elif relation == 'Explicit':
                conn1 = row['Connective_RawText']
            else:
                raise NotImplementedError(f'cannot find conn for this relation type: {relation}')
            class_set = set()

            for conn_class_str in (
                    row['ConnHeadSemClass1'], row['ConnHeadSemClass2'], row['Conn2SemClass1'], row['Conn2SemClass2']):
                if conn_class_str == '':
                    continue
                class_id = rel_map[conn_class_str]
                if class_id == -1:
                    continue
                class_set.add(class_id)
            
            class_set = list(class_set)
            class_set=list(map(lambda x:str(x),class_set))
            if len(class_set) == 0:
                continue
            if allow_multi_rel:
                for c in class_set:
                    d = {'arg1': arg1, 'arg2': arg2, 'label': c, 'conn': conn1}
                    result_dicts.append(d)
            else:
                d = {'arg1': arg1, 'arg2': arg2, 'label': class_set[0], 'conn': conn1}
                result_dicts.append(d)       
    with open(out_file_path, 'w', encoding='utf-8', newline='') as out_file:
        fieldnames = ['arg1', 'arg2', 'label', 'conn']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames,delimiter='\t')

        writer.writerows(result_dicts)


def preprocess_pdtb_for_test(csv_file_path: str, out_file_path: str, sections: List[int],
                             num_labels: Literal[4, 11, 14],
                             relation_types: List[Literal['AltLex', 'EntRel', 'Explicit', 'Implicit', 'NoRel']],
                             allow_multi_rel: bool = True):
    if num_labels == 4:
        rel_map = rel_map_4
    elif num_labels == 11:
        rel_map = rel_map_11
    elif num_labels == 14:
        rel_map = rel_map_14
    else:
        raise "num_labels can only selected in [4,11,14]"
    with open(csv_file_path, 'r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        result_dicts = []
        for row in csv_reader:
            section = int(row['Section'])
            relation = row['Relation']
            if (section not in sections) or (relation not in relation_types):
                continue
            arg1 = row['Arg1_RawText']
            arg2 = row['Arg2_RawText']
            if relation == 'Implicit':
                conn1 = row['Conn1']
            elif relation == 'Explicit':
                conn1 = row['Connective_RawText']
            else:
                raise NotImplementedError(f'cannot find conn for this relation type: {relation}')
            class_set = set()
            for conn_class_str in (
                    row['ConnHeadSemClass1'], row['ConnHeadSemClass2'], row['Conn2SemClass1'], row['Conn2SemClass2']):
                if conn_class_str == '':
                    continue
                class_id = rel_map[conn_class_str]
                if class_id == -1:
                    continue
                class_set.add(class_id)

            class_set = list(class_set)
            class_set=list(map(lambda x:str(x),class_set))
            if len(class_set) == 0:
                continue
            d = {'arg1': arg1, 'arg2': arg2, 'label': '#'.join(class_set), 'conn': conn1}
            result_dicts.append(d)
    with open(out_file_path, 'w', encoding='utf-8', newline='') as out_file:
        fieldnames = ['arg1', 'arg2', 'label', 'conn']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames,delimiter='\t')

        writer.writerows(result_dicts)


def preprocess_data(args):
    logging.info('preprocessing data...')

    if not os.path.exists(args.dataset_dir_path):
        os.makedirs(args.dataset_dir_path)
    
    preprocess_pdtb(args.original_dataset_path, os.path.join(args.dataset_dir_path, "train.tsv"),
                             num_labels=args.num_rels,
                             relation_types=['Implicit'],
                             sections=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             allow_multi_rel=True
                             )
    
    preprocess_pdtb_for_test(args.original_dataset_path, os.path.join(args.dataset_dir_path, "dev.tsv"),
                             num_labels=args.num_rels,
                             relation_types=['Implicit'],
                             sections=[0, 1],
                             allow_multi_rel=False,
                             )
    
    preprocess_pdtb_for_test(args.original_dataset_path, os.path.join(args.dataset_dir_path, "test.tsv"),
                             num_labels=args.num_rels,
                             relation_types=['Implicit'],
                             sections=[21, 22],
                             allow_multi_rel=False,
                             )
    logging.info('done')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str,
                        help="Original dataset path")
    parser.add_argument("--dataset_dir_path", type=str,
                        help="Dataset directory path")
    parser.add_argument("--num_rels", type=int, default=4,
                        choices=[3, 4, 11, 12, 14],
                        help="how many relations are computed")
    args = parser.parse_args()
    preprocess_data(args)
