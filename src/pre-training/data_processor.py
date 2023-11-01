import json
import math
import random
random.seed(42)
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


class GigaProcessor():
    def __init__(self, args, tokenizer):
        self.args = args
        self.raw_lines = []
        self.train_path = os.path.join(self.args.path_datasets, 'train.json')
        self.test_path = os.path.join(self.args.path_datasets, 'test.json')
        self.tokenizer = tokenizer
        
        self.train_raw_data = self.open_file(self.train_path)
        self.test_raw_data = self.open_file(self.test_path)
        self.vocab_len = len(self.tokenizer.get_vocab())
        self.special_token_id = [self.tokenizer.mask_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id,
                                 self.tokenizer.pad_token_id]

    def get_data_loader(self, type):
        if type == 'train':
            input_ids, attention_mask, token_type_ids, labels, conns_index  = self.mask_data(self.train_raw_data, type)
        else:
            input_ids, attention_mask, token_type_ids, labels, conns_index  = self.mask_data(self.test_raw_data, type)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        conns_index = torch.tensor(conns_index, dtype=torch.long)
        datasets = TensorDataset(input_ids, attention_mask, token_type_ids, labels, conns_index)
        if type == 'train':
            data_loader = DataLoader(datasets, shuffle = True, batch_size=self.args.batch_size)
        else:
            data_loader = DataLoader(datasets, shuffle = False, batch_size=self.args.batch_size)
        return data_loader


    def mask_data(self, data, type):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        conns_index = []
        for d in data:
            format_data = self.mask_single_data(d)
            input_ids.append(format_data["mask_input_ids"])
            attention_mask.append(format_data["attention_mask"])
            token_type_ids.append(format_data["token_type_ids"])
            labels.append(format_data["label"])
            conns_index.append(format_data["conn_index"])
               
        return input_ids, attention_mask, token_type_ids, labels, conns_index

    def mask_single_data(self, sent):
        sent_text = sent['text']
        sent_conn_index = sent['conn_index']
        sent_encode_token = self.tokenizer(sent_text, truncation=True, max_length=self.args.sen_max_length, padding='max_length')
        label = [-100 for x in sent_encode_token['input_ids']]
        attention_mask = sent_encode_token['attention_mask']
        token_type_ids = [0 for x in sent_encode_token['input_ids']]
        raw_input_ids = [x for x in sent_encode_token['input_ids']]

        if self.args.connective_mask:
            label[sent_conn_index] = raw_input_ids[sent_conn_index]
            if random.random() < 0.9: 
                raw_input_ids[sent_conn_index] = self.tokenizer.mask_token_id
            
        if self.args.mlm:
            raw_input_ids, label = self.mlm(raw_input_ids, sent_conn_index, label) 
        
        format_data = {
            "mask_input_ids": raw_input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": label,
            "conn_index": [sent_conn_index],
        }
        return format_data
        
    def open_file(self, path):
        with open(path) as f:
            data_lines = json.load(f)
        return data_lines


    def mlm(self, encode_token, conn_index, label):
        for i in range(0, len(encode_token)):
            if i == conn_index:
                continue
            encode_token[i], label[i] = self.op_mask(encode_token[i])
        return encode_token, label
    
    def op_mask(self, token):
        """
        The original mask mechanism of Bert.
        (1) With an 85% probability, the original word remains unchanged.
        (2) With a 15% probability, replacement occurs as follows:
            - 80% probability: the token is replaced with [MASK].
            - 10% probability: a token randomly selected from the vocabulary replaces the current token.
            - 10% probability: the original word remains unchanged.
        """
        if token in self.special_token_id:
            return token, -100

        if random.random() <= 0.15:
            x = random.random()
            label = token
            if x <= 0.80:
                token = self.tokenizer.mask_token_id
            if x > 0.80 and x <= 0.9:
                while True:
                    token = random.randint(0, self.vocab_len - 1)
                    if token not in self.special_token_id:
                        break

            return token, label
        return token, -100
    

    def get_len(self):
        return math.ceil((len(self.train_raw_data)) / (self.args.batch_size))




