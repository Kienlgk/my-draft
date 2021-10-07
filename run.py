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


This approach trains and uses pair of paragraph and code snippet on a Stack Overflow thread to classify if the 
Text-Code embedding and the API doc-impl embedding si possitive

Working on training and using only the CLS token to disambiguate
"""

from __future__ import absolute_import
import os
import sys
import math
import traceback
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from copy import deepcopy
from io import open
from itertools import cycle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import torch.nn as nn
from model import Encoder, SimilarityClassifier
from utils import EarlyStopping



from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,WeightedRandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from transformers import AutoTokenizer, AutoModel

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from dataset import TripletTensorDataset, TripletTensorDatasetV2, TripletTensorDatasetV3
# from transformers import RobertaForMaskedLM as RobertaModel
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Pair(object):
    """A single text-code (nl-pl) pair."""
    def __init__(self,
                 idx,
                 simple_name,
                 thread_id,
                 text,
                 code
                 ):
        self.idx = idx
        self.simple_name = simple_name
        self.thread_id = thread_id
        self.text = text
        self.code = code

class LabeledPair(object):
    """A single text-code (nl-pl) pair."""
    def __init__(self,
                 idx,
                 simple_name,
                 thread_id,
                 text,
                 code,
                 api_comment,
                 api_implementation,
                 label
                 ):
        self.idx = idx
        self.simple_name = simple_name
        self.thread_id = thread_id
        self.text = text
        self.code = code
        self.cmt = api_comment
        self.impl = api_implementation
        self.label = label

def read_pairs(filename, mode="infer"):
    pairs = []
    if mode == "infer":
        with open(filename, "r") as fp:
            nl_pl_pairs = fp.readlines()
        for pair_info in nl_pl_pairs:
            pair_info = json.loads(pair_info)
            text, code = pair_info['pairs']
            pairs.append(
                Pair(
                    pair_info['idx'],
                    pair_info['simple_name'],
                    pair_info['thread_id'],
                    text,
                    code
                    )
            )
    elif mode == "train" or mode == "test" or mode == "eval":
        with open(filename, "r") as fp:
            training_pairs = fp.readlines()
        for training_info in training_pairs:
            training_info = json.loads(training_info)
            text, code = training_info['pairs']
            api_comment, api_implementation = training_info['doc_impl']
            label = training_info['cls_label']
            pairs.append(
                LabeledPair(
                    training_info['idx'],
                    training_info['simple_name'],
                    training_info['thread_id'],
                    text,
                    code,
                    api_comment,
                    api_implementation,
                    label
                )
            )
    return pairs

class PairFeatures(object):
    """
        A pair features combining text and code into source.
        For inference mode.
    """
    def __init__(self,
                 pair_id,
                 source_ids,
                 source_mask
                ):
        self.pair_id = pair_id
        self.source_ids = source_ids
        self.source_mask = source_mask

class LabeledPairFeatures(object):
    """
        A pair features combining text and code into source
        Has label. For training, testing and evaluating mode.
    """
    def __init__(self,
                 pair_id,
                 text_code_ids,
                 text_code_mask,
                 cmt_impl_ids,
                 cmt_impl_mask,
                 label
                ):
        self.pair_id = pair_id
        self.text_code_ids = text_code_ids
        self.text_code_mask = text_code_mask
        self.cmt_impl_ids = cmt_impl_ids
        self.cmt_impl_mask = cmt_impl_mask
        self.label = label

def truncate_pair(longer_one, shorter_one, args=None):
    total_length = len(longer_one) + len(shorter_one)
    delta = len(longer_one) - len(shorter_one)
    if len(shorter_one) < 200:
        # tokenized_test = tokenized_test[:-1*delta]
        longer_one = longer_one[:-1*(total_length-args.max_source_length+3)]
    else:
        longer_one = longer_one[:-1*delta]
        new_total_length = len(longer_one) + len(shorter_one)
        need_remove_tokens_length = new_total_length - args.max_source_length+3
        if need_remove_tokens_length % 2 == 0:
            longer_one = longer_one[:-1*int(need_remove_tokens_length/2)]
            shorter_one = shorter_one[:-1*int(need_remove_tokens_length/2)]
        else:
            longer_one = longer_one[:-1*int(need_remove_tokens_length/2+1)]
            shorter_one = shorter_one[:-1*int(need_remove_tokens_length/2)]
    return longer_one, shorter_one

def convert_pairs_to_features(pairs, tokenizer, args=None):
    features = []
    for pair_i, pair in enumerate(pairs):
        tokenized_text = tokenizer.tokenize(pair.text)
        tokenized_code = tokenizer.tokenize(pair.code)
        # truncate
        total_length = len(tokenized_text) + len(tokenized_code)
        if total_length > args.max_source_length -3:
            if len(tokenized_text) > len(tokenized_code):
                tokenized_text, tokenized_code = truncate_pair(tokenized_text, tokenized_code, args)
            else:
                tokenized_code, tokenized_text = truncate_pair(tokenized_code, tokenized_text, args)

        source_tokens = [tokenizer.cls_token]+tokenized_text+[tokenizer.sep_token]+tokenized_code+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
        features.append(
            PairFeatures(
                pair.idx,
                source_ids,
                source_mask
            )
        )
    return features

def get_training_features(pairs, tokenizer, args=None):
    def truncate(tokenized_text, tokenized_code):
        total_length = len(tokenized_text) + len(tokenized_code)
        if total_length > args.max_source_length -3:
            if len(tokenized_text) > len(tokenized_code):
                tokenized_text, tokenized_code = truncate_pair(tokenized_text, tokenized_code, args)
            else:
                tokenized_code, tokenized_text = truncate_pair(tokenized_code, tokenized_text, args)

        source_tokens = [tokenizer.cls_token]+tokenized_text+[tokenizer.sep_token]+tokenized_code+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
        return source_ids, source_mask

    features = []
    nrof_positives = 0
    nrof_negatives = 0

    for pair_i, pair in enumerate(pairs):
        tokenized_text = tokenizer.tokenize(pair.text)
        tokenized_code = tokenizer.tokenize(pair.code)
        tokenized_cmt = tokenizer.tokenize(pair.cmt)
        tokenized_impl = tokenizer.tokenize(pair.impl)
        # label = [0, 1] if pair.label else [1, 0] #binary classification if text_code is relevant to cmt_impl
        label = 1 if pair.label else 0 # For nn.CrossEntropyLoss label
        if label == 1:
            nrof_positives += 1
        else:
            nrof_negatives += 1

        # truncate
        text_code_ids, text_code_mask = truncate(tokenized_text, tokenized_code)
        cmt_impl_ids, cmt_impl_mask = truncate(tokenized_cmt, tokenized_impl)

        features.append(
            LabeledPairFeatures(
                pair.idx,
                text_code_ids,
                text_code_mask,
                cmt_impl_ids,
                cmt_impl_mask,
                label
            )
        )
    return features, nrof_positives, nrof_negatives

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    


    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action='store_true',
                        help="Whether to run inferencing on the new files.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    args.output_dir = os.path.join(args.output_dir, "0")
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    # code2nl tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("{ input }")) )
    # source_ids =  tokenizer.convert_tokens_to_ids(tokenizer.tokenize("{ input }")) 
    
    # exit()
    # build model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    
    # with open("CodeBERT_arch.txt", "w+") as fp:
    #     print("[CodeBERT Encoder Architecture]", file=fp)
    #     print(str(encoder), file=fp)

    model= SimilarityClassifier(encoder=encoder,config=config, max_length=args.max_target_length,
                                sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        torch.cuda.memory_summary()
        model.load_state_dict(torch.load(args.load_model_path,map_location=args.device))
        # model.load_state_dict(torch.load(args.load_model_path, map_location='cuda:1'))
    model.to(device)

        
    
    # with open("dnn_arch.txt", "a") as fp:
    #         print("[Full DNN Architecture]", file=fp)
    #         print(str(model), file=fp)

    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)


    def assign_token_indice(placeholder, list_subwords_indices):
        # f.token_subwords_indices
        subword_indices_vector = deepcopy(placeholder)
        try:
            subword_indices_vector[:len(list_subwords_indices)] = list_subwords_indices
        except Exception as e:
            traceback.print_exc(e)
            exit()
        return subword_indices_vector

    if args.do_train:
        result_file = os.path.join(args.output_dir, "mod4_result_small_data.txt")
        with open(result_file, "w+") as res_fp:
            pass # to clear the old result

        train_file = "data/approach_4/text_code_pairs_train.jsonl"
        train_pairs = read_pairs(args.train_filename, mode="train")
        train_features, nrof_positives, nrof_negatives = get_training_features(train_pairs, tokenizer, args)
        all_pair_id = torch.tensor([f.pair_id for f in train_features], dtype=torch.long)
        all_text_code_ids = torch.tensor([f.text_code_ids for f in train_features], dtype=torch.long)
        all_text_code_mask = torch.tensor([f.text_code_mask for f in train_features], dtype=torch.long)
        all_cmt_impl_ids = torch.tensor([f.cmt_impl_ids for f in train_features], dtype=torch.long)
        all_cmt_impl_mask = torch.tensor([f.cmt_impl_mask for f in train_features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        labels = np.array([f.label for f in train_features])
        print("labels.shape", labels.shape)
        labels = labels.astype(int)

        num_of_majority_class_training_examples = nrof_negatives
        num_of_minority_class_training_examples = nrof_positives
        majority_weight = 1/num_of_majority_class_training_examples
        minority_weight = 1.5/num_of_minority_class_training_examples
        sample_weights = np.array([majority_weight, minority_weight]) # minority has integer 1 as label
        weights = sample_weights[labels]

        train_dataset = TensorDataset(all_pair_id, 
                                      all_text_code_ids, 
                                      all_text_code_mask, 
                                      all_cmt_impl_ids, 
                                      all_cmt_impl_mask, 
                                      all_label)
        
        # train_sampler = RandomSampler(train_dataset)
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        early_stopping = EarlyStopping()
        citeration = nn.CrossEntropyLoss()
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_pairs))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_pairs))

        model.train()

        # Freeze the parameters of CodeBERT
        for child_i, child in enumerate(model.encoder.children()):
            for k,v in child.named_parameters():
                if child_i == 0:
                    v.requires_grad = False
        

        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_acc,best_loss = 0, 0,0,0,0,1e6 
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        eval_flag = True
        first_eval = True
        for step in bar:
            # pair_id, text_code_ids, text_code_mask, cmt_impl_ids, cmt_impl_mask, label = next(train_dataloader)
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            pair_id, text_code_ids, text_code_mask, cmt_impl_ids, cmt_impl_mask, label = batch

            with autocast():
                preds = model(text_code_ids, text_code_mask, cmt_impl_ids, cmt_impl_mask)
                loss = citeration(preds, label)

           
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            skip_updating = False
            if not math.isnan(loss):
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("loss {}".format(train_loss))
                nb_tr_examples += pair_id.size(0)
                nb_tr_steps += 1
                skip_updating =False
                try:
                    loss.backward()
                except Exception as e:
                    skip_updating = True
            else:
                skip_updating = True


            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                eval_flag = True
                #Update parameters
                if  not skip_updating:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            if skip_updating:
                optimizer.zero_grad()
                skip_updating = False

            
            # if args.do_eval and eval_flag and (global_step+1)% args.eval_steps == 0:
            # do eval
            e_steps = 1000
            if args.do_eval and eval_flag and ((global_step == 1) or (global_step+1)% e_steps == 0):
                eval_pairs = read_pairs(args.dev_filename, mode="train")
                eval_features, _, _ = get_training_features(eval_pairs, tokenizer, args)
                eval_pair_id = torch.tensor([f.pair_id for f in eval_features], dtype=torch.long)
                eval_text_code_ids = torch.tensor([f.text_code_ids for f in eval_features], dtype=torch.long)
                eval_text_code_mask = torch.tensor([f.text_code_mask for f in eval_features], dtype=torch.long)
                eval_cmt_impl_ids = torch.tensor([f.cmt_impl_ids for f in eval_features], dtype=torch.long)
                eval_cmt_impl_mask = torch.tensor([f.cmt_impl_mask for f in eval_features], dtype=torch.long)
                eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                
                eval_dataset = TensorDataset(
                                            eval_pair_id, 
                                            eval_text_code_ids, 
                                            eval_text_code_mask, 
                                            eval_cmt_impl_ids, 
                                            eval_cmt_impl_mask, 
                                            eval_label)

                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_pairs))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                count = 0
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                softmax = nn.Softmax(dim=1)
                for eval_batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                    eval_batch = tuple(t.to(device) for t in eval_batch)
                    eval_pair_id, eval_text_code_ids, eval_text_code_mask, eval_cmt_impl_ids, eval_cmt_impl_mask, eval_label = eval_batch
                    eval_preds = model(
                                    eval_text_code_ids, 
                                    eval_text_code_mask, 
                                    eval_cmt_impl_ids, 
                                    eval_cmt_impl_mask
                                    )
                    eval_preds = softmax(eval_preds)
                    # print(eval_preds.cpu().detach().numpy())
                    # print(eval_label)
                    chosen_preds = torch.argmax(eval_preds, dim=1)
                    # print(chosen_preds.cpu().detach().numpy())
                    for eval_i, each_label in enumerate(eval_label):
                        eval_predict_result = chosen_preds[eval_i].cpu().detach().numpy()
                        # print("each_label", each_label.item())
                        # print("eval_predict_result", eval_predict_result)
                        if each_label == 1: # positive
                            if eval_predict_result == each_label.item():
                                tp += 1
                            else:
                                fn += 1
                        else:
                            if eval_predict_result == each_label.item():
                                tn += 1
                            else:
                                fp += 1
                
                prec = tp/(tp+fp)
                if tp+fn == 0:
                    if tn > 0:
                        recall = 1
                    else:
                        recall = 0
                else:
                    recall = tp/(tp+fn)
                f1 = 2*tp/(2*tp+fp+fn)
                model.train()
                # eval_loss = eval_loss
                result = {#'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5),
                        #   'eval_prec': round(prec, 5),
                        #   'eval_recall': round(recall, 5),
                          'eval_f1': round(f1, 5),
                          'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
                with open(result_file, "a") as res_fp:
                    print(json.dumps(result), file=res_fp)

                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
                epoch_no = round((global_step+1)/ (math.ceil(len(train_pairs)/args.train_batch_size)))
                logger.info(f"Eval at epoch {epoch_no}")
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                # output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                output_model_file = os.path.join(last_output_dir, f"pytorch_model_{epoch_no:04d}.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info(f"Saved last model at: {output_model_file}")

                # if eval_loss<best_loss:
                if (first_eval):
                    first_eval = False
                    continue
                if f1 > best_acc:
                    # logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info(f"Best acc: {round(f1, 4)}")
                    logger.info("  "+"*"*20)
                    # best_loss=eval_loss
                    best_acc=f1
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model_best.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info(f"Saved best model at: {output_model_file}")

                early_stopping(f1)
                if early_stopping.early_stop:
                    logger.info(f"Early stop at global steps: {global_step+1}")
                    break


                    

        
        pass
    if args.do_test:
        pass
    if args.do_infer:
        activation = {}

        for child_i, child in enumerate(model.children()):
            for k,v in child.named_parameters():
                v.requires_grad = False

        files=[]
        infer_out_dir = "data/approach_4/"
        os.makedirs(infer_out_dir, exist_ok=True)

        # Stack Overflow pairs
        infer_file="data/approach_4/text_code_pairs_test.jsonl"
        # infer_file="data/approach_4/sample_train_pairs.jsonl"
        file_path = infer_out_dir+os.sep+"test_127_threads_result.json"

        # API doc implementation pairs
        # infer_file="data/approach_4/doc_impl_pairs.jsonl"
        # file_path = infer_out_dir+os.sep+"api_doc_impl_embs.json"


        eval_examples = read_pairs(infer_file, mode="train")
        # eval_features = convert_pairs_to_features(eval_examples, tokenizer, args)
        eval_features, _, _ = get_training_features(eval_examples, tokenizer, args)
        eval_pair_id = torch.tensor([f.pair_id for f in eval_features], dtype=torch.long)
        eval_text_code_ids = torch.tensor([f.text_code_ids for f in eval_features], dtype=torch.long)
        eval_text_code_mask = torch.tensor([f.text_code_mask for f in eval_features], dtype=torch.long)
        eval_cmt_impl_ids = torch.tensor([f.cmt_impl_ids for f in eval_features], dtype=torch.long)
        eval_cmt_impl_mask = torch.tensor([f.cmt_impl_mask for f in eval_features], dtype=torch.long)
        eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

        eval_dataset = TensorDataset(
                                    eval_pair_id, 
                                    eval_text_code_ids, 
                                    eval_text_code_mask, 
                                    eval_cmt_impl_ids, 
                                    eval_cmt_impl_mask, 
                                    eval_label)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

        logger.info("\n***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        result_idx = 0
        output_results = {}
        softmax = nn.Softmax(dim=1)
        for eval_batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            eval_batch = tuple(t.to(device) for t in eval_batch)
            eval_pair_id, eval_text_code_ids, eval_text_code_mask, eval_cmt_impl_ids, eval_cmt_impl_mask, eval_label = eval_batch
            eval_preds = model(
                            eval_text_code_ids, 
                            eval_text_code_mask, 
                            eval_cmt_impl_ids, 
                            eval_cmt_impl_mask
                            )
            
            eval_preds = softmax(eval_preds)
            chosen_preds = torch.argmax(eval_preds, dim=1)
            for eval_i, each_label in enumerate(eval_label):
                eval_predict_result = chosen_preds[eval_i].cpu().detach().numpy()
                output_results[str(result_idx)] = eval_predict_result.tolist()
                result_idx += 1
        
        import codecs
        with open(file_path, "w+") as fp:
            json.dump(output_results, fp, indent=2)
if __name__ == "__main__":
    main()


