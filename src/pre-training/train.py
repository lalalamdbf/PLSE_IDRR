import os
import time
import logging
import math
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import  AdamW, get_cosine_schedule_with_warmup
from transformers import RobertaForMaskedLM
from PromptModel import RobertaForPrompt
from transformers import RobertaTokenizer
from data_processor import GigaProcessor
import json


def train(args, logger):

    logger.info('training start')
    save_config(args, logger)
    device = torch.device('cuda')

    # data loading
    logger.info('data loading')
    tokenizer = RobertaTokenizer.from_pretrained(args.initial_pretrain_model)
    data_processor = GigaProcessor(args, tokenizer)
    

    #  Initialize the model 
    logger.info('model loading')
    if args.lse:
        model = RobertaForPrompt.from_pretrained(args.initial_pretrain_model)
    else:
        model = RobertaForMaskedLM.from_pretrained(args.initial_pretrain_model)

    
    state_dict = None
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    num_training_steps = args.num_epochs * data_processor.get_len()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_ratio * num_training_steps,
        num_training_steps=num_training_steps
    )
    
    # Distributed training
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    logger.info('start to train')
    model.train()
    
    loss_best = math.inf

    start_epoch = 0
    end_epoch = args.num_epochs  
    cur_training_steps = end_epoch * data_processor.get_len()
        
    progress_bar = tqdm(range(cur_training_steps))
    eval_dl = data_processor.get_data_loader('test')

    for epoch in range(start_epoch, end_epoch):
        train_dl = data_processor.get_data_loader('train')
        logger.info("Load datalaoder Finished")
        for i, batch in enumerate(train_dl):
            input_ids, attention_mask, token_type_ids, labels, conns_index = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            conns_index = conns_index.to(device)

            if args.lse:
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,conns_index=conns_index)
            else:
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            
            mask_loss = outputs.loss
            if args.lse:
                mutual_loss = outputs.mutual_loss
            if torch.cuda.device_count() > 1:
                mask_loss = mask_loss.mean()
                if args.lse:
                    mutual_loss = mutual_loss.mean()
                
            if args.lse:
                loss = mask_loss + mutual_loss
            else:
                loss = mask_loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if i % 250 == 0:
                if args.lse:
                    logger.info('epoch:{0}  iter:{1}/{2}  loss:{3}  mutual_loss:{4}  mask_loss:{5}'.format(epoch, i, len(train_dl), loss.item(), mutual_loss.item(),  mask_loss.item()))
                else:
                    logger.info('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_dl), loss.item()))

        current_loss = eval(eval_dl, args, model, epoch, logger, device)

        if current_loss < loss_best:
            loss_best = current_loss
            logger.info('saving model')
            path = args.path_model_save + 'pretrain_state_epoch_{}.ckpt'.format(epoch)
  
            model_save = model.module if torch.cuda.device_count() > 1 else model
            state_dict = {
                'epoch': epoch,
                'net': model_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr_scheduler.state_dict()
            }
            torch.save(state_dict, path)
            logger.info(f"Save epoch-{epoch} model state")
            
def eval(eval_dataloader, args, model, epoch, logger, device):
    logger.info(f"start epoch_{epoch} test")
    mutual_losses = []
    mask_losses = []
    losses = []
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        input_ids, attention_mask, token_type_ids, labels, conns_index = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        conns_index = conns_index.to(device)
        with torch.no_grad():
            if args.lse:
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels, conns_index=conns_index)
            else:
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        
        mask_loss = outputs.loss
        if args.lse:
            mutual_loss = outputs.mutual_loss
        
        if torch.cuda.device_count() > 1:
            mask_loss = mask_loss.mean()
            if args.lse:
                mutual_loss = mutual_loss.mean()
    
        if args.lse:
            loss = mask_loss + mutual_loss
        else:
            loss =  mask_loss
        
        if args.lse:
            mutual_losses.append(mutual_loss)
            mask_losses.append(mask_loss)
        losses.append(loss)
        
    if args.lse:
        mutual_losses = torch.tensor(mutual_losses, device)
        mask_losses = torch.tensor(mask_losses, device)
    losses = torch.tensor(losses, device)
    
    if args.lse:
        mutual_losses_avg = torch.mean(mutual_losses)
        mask_losses_avg = torch.mean(mask_losses)
    losses_avg = torch.mean(losses)
    if args.lse:
        logger.info('eval {0}: loss:{1}  mutual_loss:{2}  mask_loss:{3}'.format(epoch, losses_avg.item(), mutual_losses_avg.item(), mask_losses_avg.item()))
    else:
        logger.info('eval {0}: loss:{1}'.format(epoch, losses_avg.item()))
    return losses_avg

def save_config(args, logger):
    logger.info("save config")
    run_conf = {
        'lr' : args.learning_rate,
        'batch_size': args.batch_size,
        'max_length': args.sen_max_length,
        'mlm': args.mlm,
        'lse': args.lse,
        'connective_mask': args.connective_mask,
    }
    json.dump(run_conf,open(os.path.join(args.path_model_save,"train_config.json"),'w'),ensure_ascii=False,indent=4)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--num_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--sen_max_length", default=256, type=int)
    parser.add_argument("--initial_pretrain_model", default='', type=str)
    parser.add_argument("--path_model_save", default='', type=str)
    parser.add_argument("--path_datasets", default='', type=str)
    parser.add_argument("--path_log", default='', type=str)
    parser.add_argument("--lse", action="store_true",
                        help="global logical semantics enhancement")
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--connective_mask", action="store_true")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    path_log = os.path.join(args.path_model_save,'logs/')
    if not os.path.exists(args.path_model_save): 
            os.mkdir(args.path_model_save)
    if not os.path.exists(path_log):
        os.mkdir(path_log)
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(path_log,f"pretrain_{time.time()}.log"), 'a')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    train(args, logger)