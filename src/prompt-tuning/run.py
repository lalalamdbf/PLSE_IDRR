import os
import shutil
import time
import torch
import argparse
import logging
from data_process import PromptDataGenerate
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange
import numpy as np
from prompt_model import PromptIDRC
from evaluate import evaluate_accuracy, evaluate_precision_recall_f1
from torch.optim import AdamW
import json
from prompt_config import PromptConfig

def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        try:
            x = eval(x)
        except:
            pass
        results.append(x)
    return results

def print_model_result(result, data_type='train'):
    for key in sorted(result.keys()):
        print(" \t %s = %-5.5f" % (key, float(result[key])), end="")



def model_eval(model, args, data_loader, data_type='dev', epoch_num=-1, metric=None, device=None):
    result_sum = {}
    nm_batch = 0
    labels_pred = list()
    multi_labels_true = list()
    total_cnt = 0
    total_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader)):

        labels = batch.label
        bsz = len(labels)
        total_cnt += bsz

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token_type_ids = batch.token_type_ids
        loss_ids = batch.loss_ids
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        loss_ids = loss_ids.to(device)
        label = labels.to(device)
        model.eval()
        with torch.no_grad():
            loss, pred = model(input_ids, attention_mask, token_type_ids, loss_ids, label)
        
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            
        total_loss += loss.item() * bsz
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        labels_pred.extend(pred.tolist())
        multi_labels_true.extend(labels.tolist())

        nm_batch += 1

    result_sum["loss"] = total_loss / total_cnt
    result_sum["accuracy"] = evaluate_accuracy(np.array(labels_pred), np.array(multi_labels_true))
    f1_results = evaluate_precision_recall_f1(np.array(labels_pred), np.array(multi_labels_true))
    result_sum["f1-detail"] = f1_results
    result_sum["f1"] = f1_results["overall"][-1]
    print(result_sum["f1"])
    with open(os.path.join(args.output_dir, 'Discourage_' + data_type + '_result.txt'), 'a+',
              encoding='utf-8') as writer:
        print("***** Eval results in " + data_type + "*****")
        if data_type == 'dev':
            writer.write(f"======{data_type} Epoch {epoch_num}=========\n")
        else:
            writer.write(f"======{data_type} Best {metric}=========\n")
        for key in sorted(result_sum.keys()):
            print("%s = %s" % (key, str(result_sum[key])))
            writer.write("%s = %s\n" % (key, str(result_sum[key])))
        writer.write('\n')

    return result_sum

def save_best_model(model, args, v, optimizer=None, data_type='dev', eval_best = None, train_best = None, use_f1=False, logger = None):
     # using loss as the evaluation metric
    if not use_f1 and data_type == 'dev':
        if eval_best > v:
            eval_best = v
            state = {
                'prompt_model': model.prompt_model.state_dict(),
                'optimizer': optimizer
            }
            save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_' +
                                 data_type + '_loss_' + str(v) + '.model')
            torch.save(state, save_path)
            logger.info(f"========Save best loss model {save_path}=========\n")
            train_best = save_path
        return eval_best, train_best

    # using F1 score as the evaluation metric
    if use_f1 and data_type == 'dev':
        if eval_best < v:
            eval_best = v
            state = {
                'prompt_model': model.prompt_model.state_dict(),
                'optimizer': optimizer
            }
            save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_'
                                     + data_type + '_f1_' + str(v) + '.model')
            torch.save(state, save_path)
            logger.info(f"========Save best f1 model {save_path}=========\n")
            train_best = save_path
        return eval_best, train_best

def save_epoch_model(model, epoch, args, logger):
    model.eval()
    state = {'prompt_model': model.prompt_model.state_dict()}
    save_path = os.path.join(args.output_dir, 'Discourage' + '_state_dict_' + '_epoch_' + str(epoch) + '.model')
    logger.info(f"Save Epoch {epoch} Model\n")
    torch.save(state, save_path)

def save_config(args, prompt_config, logger):
    logger.info("save config")
    run_conf = {
        'lr' : args.learning_rate,
        'train_batch_size':args.train_batch_size,
        'template': prompt_config.get_template().text,
        'verbalizer': prompt_config.get_verbalizer().label_words,
        'use_pretrain': args.use_pretrain,
        'pretrain_file': args.plse_pretrain_file
    }
    json.dump(run_conf,open(os.path.join(args.output_dir,"train_config.json"),'w'),ensure_ascii=False,indent=4)
def inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model
def train(model, args, data_generator, prompt_config, logger,  device):
    
    dev_loader = data_generator.get_dev_loader()
    train_loader = data_generator.get_train_loader()
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)

    global_step = 0
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    save_config(args, prompt_config, logger)

    eval_best_loss = 999
    eval_best_f1 = 0
    train_best_loss_model = None
    train_best_f1_model = None
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            
            labels = batch.label
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            loss_ids = batch.loss_ids
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            loss_ids = loss_ids.to(device)
            label = labels.to(device)

            loss, output = model(input_ids, attention_mask, token_type_ids, loss_ids, label)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if args.do_eval:
            model.eval()
            print("\nepoch:{} global:{}\t".format(epoch, global_step))
            eval_result = model_eval(inner_model(model), args, dev_loader, data_type='dev',epoch_num=epoch, metric=None, device=device)

            # save the model using the loss as the evaluation criterion
            eval_best_loss, train_best_loss_model = save_best_model(inner_model(model), args, eval_result['loss'], optimizer.state_dict(), eval_best = eval_best_loss, data_type='dev', logger=logger)
            # save the model using F1 score as the evaluation criterion
            eval_best_f1, train_best_f1_model = save_best_model(inner_model(model), args, eval_result['f1'], optimizer.state_dict(), eval_best = eval_best_f1, data_type='dev',
                                use_f1=True, logger=logger)
        save_epoch_model(inner_model(model), epoch, args, logger)

    shutil.copy(train_best_f1_model, os.path.join(args.output_dir, 'best_f1_model.bin'))
    shutil.copy(train_best_loss_model, os.path.join(args.output_dir, 'best_loss_model.bin'))


def eval_test(model, args, data_generator, logger, device):
    model.eval()
    best_model_path = [os.path.join(args.output_dir, 'best_f1_model.bin'),
                       os.path.join(args.output_dir, 'best_loss_model.bin')]
    for best_model in best_model_path:
        checkpoint = torch.load(best_model)['prompt_model']
        model.prompt_model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        test_loader = data_generator.get_test_loader()
        
        logger.info("\n********" + best_model + "********")
        model_eval(model, args, test_loader, data_type='test', metric = 'loss' if 'loss' in best_model else 'f1', device=device)
        if args.num_rels == 14:
            blind_loader = data_generator.get_blind_loader()
            model_eval(model, args, blind_loader, data_type='blind', metric = 'loss' if 'loss' in best_model else 'f1', device=device)
    pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_ids", type=str2list, default=None,
                        help="gpu ids") 
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="number of train epochs") 
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--dev_batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--pretrain_file", default='', type=str)
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument("--plse_pretrain_file", default='', type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--num_rels", type=int, default=4,
                        choices=[4, 11, 14],
                        help="how many relations are computed")
    parser.add_argument("--train_best_f1_model", default='', type=str)
    parser.add_argument("--train_best_loss_model", default='', type=str)
    parser.add_argument("--do_train", action="store_true") 
    parser.add_argument("--do_eval", action="store_true") 
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_pretrain", action="store_true")
    
    args = parser.parse_args()
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(args.output_dir, "Prompt.log"), 'a')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    prompt_config = PromptConfig(args)
    model = PromptIDRC(prompt_config)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_generator = PromptDataGenerate(args, prompt_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = len(args.gpu_ids)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        device = torch.device("cuda:%d" % args.gpu_ids[0])
        torch.cuda.set_device(device)
        
    if args.do_train:
        train(model, args, data_generator, prompt_config, logger, device)
    if args.do_test:
        eval_test(model, args, data_generator, logger, device)
    end_time = time.time()
    print("Time Cost：%d m" % int((end_time - start_time) / 60))
    pass

