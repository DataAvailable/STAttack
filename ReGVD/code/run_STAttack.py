# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
import subprocess
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import *
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(func, label, idx, tokenizer, args):
    #source
    # code=' '.join(js['func'].split())

    code_tokens=tokenizer.tokenize(func)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,idx,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        funcs = []
        labels = []
        idxs = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                # func =' '.join(js['func'].split())
                func =' '.join(js['func'].split(" "))
                funcs.append(func)
                labels.append(js['target'])
                idxs.append(js['idx'])
                # self.examples.append(convert_examples_to_features(js, tokenizer, args))

        ######################################################   代码转化开始   ######################################################
        print("**************** 代码转化开始 ****************")
        print('********* Rules: ' + args.transformation_rules + '*********')
        transformed_funcs=[]
        transformed_labels=[]
        transformed_idxs=[]
        transdormed_index = 0
        for i in tqdm(range(len(funcs))):
            with open('temp_c.c', 'w') as f:
                f.write(funcs[i])
            subprocess.Popen("rm -rf Mutated.c", shell=True)
            subprocess.Popen("txl -q -s 128 temp_c.c ../Txl/RemoveNullStatements.Txl > ../RM/temp_code.c", shell=True)
            subprocess.Popen("txl -q ../RM/temp_code.c ../Txl/CountModification.Txl > /dev/null 2> /dev/null", shell=True)
            if args.transformation_rules == 'R1':
                # print('********* R1:重命名 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 1", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/1ChangeRename.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R2':
                # print('********* R2:常量修改 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 11", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/11changeConstant.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R3':
                # print('********* R3:算数表达 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 8", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/8changeCompoundLogicalOperator.Txl > ../RM/temp_3.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                out_3 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 9", shell=True)
                out_4 = subprocess.Popen("txl -q -s 128 ../RM/temp_3.c ../Txl/9changeSelfOperator.Txl > ../RM/temp_9.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
                out_5 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 10", shell=True)
                out_6 = subprocess.Popen("txl -q -s 128 ../RM/temp_9.c ../Txl/10changeCompoundIncrement.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R4':
                # print('********* R4:定义修改 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 12", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/12changeVariableDefinitions.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R5':
                # print('********* R5:循环改写 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 2", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/2A3ChangeCompoundForAndWhile.Txl > ../RM/temp_2.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
                out_3 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 4", shell=True)
                out_4 = subprocess.Popen("txl -q -s 128 ../RM/temp_2.c ../Txl/4changeCompoundDoWhile.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R6':
                # print('********* R6:条件改写 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 5", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/5A6changeCompoundIf.Txl > ../RM/temp_5.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
                out_3 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 7", shell=True)
                out_4 = subprocess.Popen("txl -q -s 128 ../RM/temp_5.c ../Txl/7changeCompoundSwitch.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R7':
                # print('********* R7:顺序转换 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 14", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/14changeExchangeCodeOrder.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elif args.transformation_rules == 'R8':
                # print('********* R8:冗余代码 *********')
                out_1 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 15", shell=True)
                out_2 = subprocess.Popen("txl -q -s 128 ../RM/temp_code.c ../Txl/15changeDeleteCode.Txl > ../RM/temp_15.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                out_3 = subprocess.Popen("python ../RM/GenRandomChange.py ../CountResult/ 13", shell=True)
                out_4 = subprocess.Popen("txl -q -s 128 ../RM/temp_15.c ../Txl/13changeAddJunkCode.Txl > ../RM/temp.c", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            out_str = out_2.communicate()[0].decode("utf-8")
            if args.filter_error == 'yes':
                if "Syntax error" not in out_str:
                    subprocess.Popen("cp ../RM/temp.c Mutated.c", shell=True)
                    subprocess.Popen("rm -rf ../RM/temp* ", shell=True)
                    with open('Mutated.c', 'r') as fr:
                        lines = fr.readlines()
                    func = ''.join(lines)
                    transformed_funcs.append(func)
                    transformed_labels.append(labels[i])
                    transformed_idxs.append(idxs[i])
                    transdormed_index += 1
            else:
                if "Syntax error" in out_str:
                    transformed_funcs.append(funcs[i])
                else:
                    subprocess.Popen("cp ../RM/temp.c Mutated.c", shell=True)
                    subprocess.Popen("rm -rf ../RM/temp* ", shell=True)
                    with open('Mutated.c', 'r') as fr:
                        lines = fr.readlines()
                    func = ''.join(lines)
                    transformed_funcs.append(func)
                transformed_labels.append(labels[i])
                transformed_idxs.append(idxs[i])
                transdormed_index += 1

            out_1.terminate()
            out_1.wait()
            out_2.terminate()
            out_2.wait()
            out_3.terminate()
            out_3.wait()
            out_4.terminate()
            out_4.wait()
            out_5.terminate()
            out_5.wait()
            out_6.terminate()
            # out_6.wait()
        funcs = transformed_funcs # 转化后的函数
        labels = transformed_labels # 转化后的函数对应的标签

        idxs = transformed_idxs # 转化后的函数对应的idx
        print("**************** 代码转化结束 ****************\n共转化代码数量: ", transdormed_index)
        ######################################################   代码转化结束   ######################################################


        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], idxs[i], tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            # logger.info("*** Total Sample ***")
            # logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            print("*** Total Sample ***")
            print("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                    # logger.info("*** Sample ***")
                    # logger.info("Total sample".format(idx))
                    # logger.info("idx: {}".format(idx))
                    # logger.info("label: {}".format(example.label))
                    # logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    # logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    print("*** Sample ***")
                    print("Total sample".format(idx))
                    print("idx: {}".format(idx))
                    print("label: {}".format(example.label))
                    print("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    print("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", args.max_steps)

    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
        # for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)

            # bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            # logger.info("epoch {} loss {}".format(idx, avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            # logger.info("  %s = %s", key, round(value,4))  
                            print("  %s = %s", key, round(value,4))                     
                        # Save model checkpoint
                        
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        # logger.info("  "+"*"*20)  
                        # logger.info("  Best acc:%s",round(best_acc,4))
                        # logger.info("  "+"*"*20)    

                        print("  "+"*"*20)  
                        print("  Best acc:%s",round(best_acc,4))
                        print("  "+"*"*20)                             
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        # logger.info("Saving model checkpoint to %s", output_dir)
                        print("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        # logger.info("epoch {} loss {}".format(idx, avg_loss))
        print("epoch {} loss {}".format(idx, avg_loss))
                        

def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result

def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    y_trues=[]
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        # inputs = batch[0].to(args.device)
        # label=batch[1].to(args.device) 
        (inputs, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            # labels.append(label.cpu().numpy())
            logit = model(inputs, labels)   # 模型推理/测试过程中请删除 "labels" 参数，该参数仅用于绘制 UMAP 图
            # logit = model(inputs)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    preds=logits[:,0]>0.5

    ######################### 挑选出“label为1, preds为1”的样本 #########################
    # original_pred_1_ind = [6, 9, 12, 19, 20, 22, 23, 24, 26, 29, 32, 33, 34, 40, 41, 45, 
    #                        53, 54, 56, 58, 61, 64, 67, 69, 70, 71, 75, 90, 91, 93, 95, 114, 
    #                        117, 118, 121, 124, 126, 128, 131, 132, 141, 142, 144, 162, 163, 
    #                        169, 183, 196, 198, 200, 201, 209, 216, 217, 224, 225, 243, 244, 
    #                        248, 249, 252, 253, 262, 264, 274, 279, 284, 286, 287, 292, 294, 
    #                        295, 297, 298, 304, 311, 316, 321, 325, 326, 332, 337, 343, 345,
    #                        347, 351, 352, 354, 356, 358, 359, 376, 386, 389, 390, 391, 393,
    #                        396]
    # index_1 = []
    # for i in range(len(y_trues)):     
    #     if y_trues[i] == 1 and preds[i] == False:
    #         index_1.append(i)
    # index_label_1_pred_0 = []                       # [9, 34, 67, 71, 141, 144, 169, 183, 196, 198]
    # for index in index_1:                           # [24, 34, 61, 196]
    #     for original in original_pred_1_ind:        # [9, 61, 90, 141, 144, 169, 183, 196]
    #         if index == original:                   # [9, 34, 75, 124, 141, 183, 196]
    #             index_label_1_pred_0.append(index)  # [33, 34, 75, 124, 128, 163, 169, 183, 196]
    ######################### 挑选出“label为1, preds为1”的样本 #########################

    accuracy = accuracy_score(y_trues, preds)
    recall = recall_score(y_trues, preds)
    precision = precision_score(y_trues, preds)
    f1 = f1_score(y_trues, preds)   

    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,preds):
            if pred:
                f.write(example.idx+'\t1\n')
            else:
                f.write(example.idx+'\t0\n')

    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
    }
    return result
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    parser.add_argument("--model", default="GNNs", type=str,help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")

    parser.add_argument("--format", default="uni", type=str, help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percet of training sample")

    parser.add_argument("--transformation_rules", default="R1", type=str, help="select the transformation rules (R1,...,R8)")
    parser.add_argument("--filter_error", default="yes", type=str, help="whether filters the codes that are compiled failed!")


    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//max(args.n_gpu,1)
    args.per_gpu_eval_batch_size=args.eval_batch_size//max(args.n_gpu,1)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    print("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        # logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))
        print("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    ## model
    # if args.model == "original":
    #     model = Model(model,config,tokenizer,args)
    if args.model == "devign":
        model = DevignModel(model, config, tokenizer, args)
    else: #GNNs
        model = GNNReGVD(model, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # logger.info("Training/evaluation parameters %s", args)
    print("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result=evaluate(args, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test_result = test(args, model, tokenizer)

            logger.info("***** Test results *****")
            for key in sorted(test_result.keys()):
                logger.info("  %s = %s", key, str(round(test_result[key],4)))

    return results


if __name__ == "__main__":
    main()


