import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
import argparse
import numpy as np
from tqdm import tqdm

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    RobertaPreTrainedModel,
    default_data_collator,
    set_seed,
    EvalPrediction,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from cl_model import ModelContra


def train(args, train_dataloader, eval_dataloader, model, tokenizer):
    """ Train the model """

    args.save_steps = len(train_dataloader) if args.save_steps<=0 else args.save_steps
    args.warmup_steps = len(train_dataloader) if args.warmup_steps<=0 else args.warmup_steps
    args.logging_steps = len(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    model.to(args.device)
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

    print("***** Running training *****")

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_eval_loss = 1e6
    model.zero_grad()

    for idx in range(int(args.num_train_epochs)):
        bar = tqdm(enumerate(train_dataloader))

        for step, batch in bar:
            batch['input_ids'] = torch.cat((batch['input_ids'][:, 0, :], batch['input_ids'][:, 1, :]), 0)
            batch['attention_mask'] = torch.cat((batch['attention_mask'][:, 0, :], batch['attention_mask'][:, 1, :]), 0)

            model.train()
            loss = model(batch)

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
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} step {} loss {}".format(idx, step+1, avg_loss))

            # logger.info("\nstep:{}, lb_sum:{}, index:{}".format(step+1, torch.sum(labels), batch[3]))
            # logger.info(", ".join([train_dataset.data[i.tolist()]['idx'] for i in batch[3]]))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        eval_loss = evaluate(args, eval_dataloader, model)
                        # Save model checkpoint
                        if best_eval_loss >= eval_loss:
                            best_eval_loss = eval_loss

                            # save
                            checkpoint_prefix = 'checkpoint-best'
                            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                            print("Saving model checkpoint to %s", output_dir)
                    if args.local_rank == -1:
                        checkpoint_prefix = 'checkpoint-last'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(output_dir)


def evaluate(args, eval_dataloader, model):
    """ eval """
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in eval_dataloader:
        batch['input_ids'] = torch.cat((batch['input_ids'][:, 0, :], batch['input_ids'][:, 1, :]), 0)
        batch['attention_mask'] = torch.cat((batch['attention_mask'][:, 0, :], batch['attention_mask'][:, 1, :]), 0)

        with torch.no_grad():
            loss = model(batch)
        nb_eval_steps += 1
        eval_loss += loss.mean().item()
    return eval_loss / nb_eval_steps


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity.")
    parser.add_argument("--pt_model", default="roberta", type=str,
                        help="The model used for contrastive learning pretraining.")
    parser.add_argument("--encoder_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="The checkpoint path of model to continue training.")

    parser.add_argument("--config_name_or_path", default="", type=str,
                        help="Optional pretrained config name or path if not the same as encoder_name_or_path")
    parser.add_argument("--tokenizer_name_or_path", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as encoder_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--pad_to_max_length", action='store_true', help="Whether to pad all samples to `max_seq_length`. ")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
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
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--temp', type=float, default=0.05, help="temperature for softmax")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as encoder_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=45,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    # Load config, tokenizer and model
    config = AutoConfig.from_pretrained(args.encoder_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path, cache_dir=args.cache_dir)
    model = RobertaModel.from_pretrained(
        args.encoder_name_or_path,
        from_tf=bool(".ckpt" in args.encoder_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    # Build dataset
    data_files = {}
    data_files["train"] = args.train_data_file
    data_files["validation"] = args.eval_data_file
    extension = args.train_data_file.split(".")[-1]

    datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir, delimiter="\t" if "tsv" in args.train_data_file else ",")

    column_names = datasets["train"].column_names
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]


    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        sent_features = tokenizer(
            sentences,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features
    
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    eval_dataset = datasets["validation"].map(
        prepare_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_gpu_train_batch_size
    )

    model = ModelContra(model, config, tokenizer, args)

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
    
    if args.do_train:
        train(args, train_dataloader, eval_dataloader, model, tokenizer)
