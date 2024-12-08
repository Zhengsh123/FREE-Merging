import copy
import os
import sys
import argparse
from functools import partial
import time
import logging
import json
import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from collections import OrderedDict
from utils_file.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils_file.metrics import compute_metrics
from utils_file.customized_trainers import CustomizedTrainer
from utils_file.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod
from inference_plms_glue import dataset_model_learning_rate_mapping_dict
from utils_file.load_config import cache_dir
from model_merging_methods.task_vector import *

parser = argparse.ArgumentParser("Interface for merging roberta models on glue")
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["roberta-base"])
parser.add_argument("--merging_method_name", type=str, default='free_merging', help="name of the method to merge models",
                    choices=["emr_merging","fr_merging","bread_merging","free_merging",'twin_merging'])
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--gpu", type=int, default=7, help="number of gpu to use")
parser.add_argument("--top_k_keep",type=float, default=0.1)
parser.add_argument("--top_k_remove",type=float, default=0.02)
parser.add_argument("--scaling",type=float, default=0.5)
parser.add_argument("--high_ratio_for_fourier",type=float, default=0.7)
args = parser.parse_args()
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

def sum_two(unied_vector, privated):
    with torch.no_grad():
        overlap_mask=(unied_vector!=0) & (privated!=0)
        new_vector= unied_vector + privated
        new_vector[overlap_mask]=privated[overlap_mask]
    return new_vector

def task_vector_param_dict_to_single_vector(task_vector):
    task_vector_param_dict = copy.deepcopy(task_vector)
    sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
    return torch.nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])


def get_emr_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)

    Vector_unified, masks, rescales = merging_method.get_emr_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = {}
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n] * masks[n][idx] * rescales[idx]
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name] + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in merged_params:
                param_value.data.copy_(merged_params[param_name])
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

def get_free_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer,top_k_keep=0.02,top_k_remove=0.00,high_ratio_for_fourier=0.7):
    logger.info(f"configuration is {args}")
    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)

    Vector_unified, private_models = merging_method.get_private_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   top_k_keep=top_k_keep,top_k_remove=top_k_remove,high_ratio_for_fourier=high_ratio_for_fourier
                                                   )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts/{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = {}
        private_model=private_models[idx].task_vector_param_dict
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n].to(args.device)*0.2 + private_model[n].to(args.device)*10
            # task_vector_recon[n]=sum_two(Vector_unified[n].to(args.device)*0.2,private_model[n].to(args.device)*10)
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name].to(args.device) + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in task_vector_recon:
                param_value.data.copy_(merged_params[param_name])
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics


def get_free_merge_performance_router(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer,top_k_keep=0.02,top_k_remove=0.00,high_ratio_for_fourier=0.7):
    logger.info(f"configuration is {args}")
    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)

    Vector_unified, private_models = merging_method.get_private_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   top_k_keep=top_k_keep,top_k_remove=top_k_remove,high_ratio_for_fourier=high_ratio_for_fourier
                                                   )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = {}
        private_model=private_models[idx].task_vector_param_dict
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n].to(args.device)*0.2 + private_model[n].to(args.device)*10
            # task_vector_recon[n]=sum_two(Vector_unified[n].to(args.device)*0.2,private_model[n].to(args.device)*10)
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name].to(args.device) + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in task_vector_recon:
                param_value.data.copy_(merged_params[param_name])
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

def get_twin_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer,top_k_keep=0.02,top_k_remove=0.00):
    logger.info(f"configuration is {args}")
    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)

    Vector_unified, private_models = merging_method.get_twin_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   top_k_keep=top_k_keep,top_k_remove=top_k_remove
                                                   )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        task_vector_recon = {}
        private_model=private_models[idx].task_vector_param_dict
        for n in Vector_unified:
            task_vector_recon[n] = Vector_unified[n].to(args.device)*0.29 + private_model[n].to(args.device)
        with torch.no_grad():
            merged_params = {}
            for param_name in task_vector_recon:
                merged_params[param_name] = pretrained_param_dict[param_name].to(args.device) + task_vector_recon[param_name]
        for param_name, param_value in merged_model.named_parameters():
            if param_name in task_vector_recon:
                param_value.data.copy_(merged_params[param_name])
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

def get_fourier_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   scaling_coefficient=args.scaling,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   high_ratio_for_fourier=args.high_ratio_for_fourier
                                                   )
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
           output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )
        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

def get_bread_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer,top_k_keep:float=0.1,top_k_remove:float=0.02):
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base").to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
    pretrained_model = copy.deepcopy(merged_model)
    pretrained_model.to('cpu')
    pretrained_param_dict = {param_name: param_value for param_name, param_value in
                             pretrained_model.named_parameters()}

    set_random_seed(seed=0)
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   scaling_coefficient=args.scaling,
                                                   top_k_keep=top_k_keep,
                                                   top_k_remove=top_k_remove
                                                   )
    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        merged_model_training_args = TrainingArguments(
            output_dir=f"./model/vanillaOVO--roberta_base_glue_ckpts{dataset_name}/{args.language_model_name}_lr{learning_rate}",
            # save model directory
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        )
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )
        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")
    return test_metrics

def mask_keep_top(top_k_keep,tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 0:
        return tensor
    else:
        top_k_int = int(tensor.shape[-1] * top_k_keep)
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
        mask = torch.zeros(tensor.shape)
        mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

        return mask * tensor


if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    assert all([dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
        'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]!'
    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        load_model_paths.append(f"./model/vanillaOVO--roberta_base_glue_ckpts{dataset_name}/{args.language_model_name}_lr{learning_rate}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="FacebookAI/roberta-base")
    except:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    # load the checkpoint of each individual model that needs to be merged
    models_to_merge, trainers, = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        # load model directory
            per_device_train_batch_size=args.batch_size,       # batch size per device during training
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        )

        assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        print(os.path.join(training_args.output_dir))
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=training_args.output_dir,
            num_labels=num_labels).to(args.device)
        trainer = CustomizedTrainer(
            model=model_to_merge,               # model to be merged
            args=training_args,                 # training arguments
            train_dataset=train_dataset,        # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        models_to_merge.append(model_to_merge.to('cpu'))
        trainers.append(trainer)
    # analysy(models_to_merge,None)
    # import sys
    # sys.exit(0)
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    save_merge_log_path = f"./save_merge_logs/{args.merging_method_name}/{args.language_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)

    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    ["emr_merging","fourier_merging","bread_merging","private_merging",'twin_merging']
    if args.merging_method_name=="emr_merging":
        performance = get_emr_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name=="fr_merging":
        performance =get_fourier_merge_performance(args,models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
    elif args.merging_method_name=="bread_merging":
        performance =get_bread_merge_performance(args,models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer,top_k_keep=args.top_k_keep,top_k_remove=args.top_k_remove)
    elif args.merging_method_name=="free_merging":
        performance =get_free_merge_performance_router(args,models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer,top_k_keep=args.top_k_keep,top_k_remove=args.top_k_remove,high_ratio_for_fourier=args.high_ratio_for_fourier)
    elif args.merging_method_name=="twin_merging":
        performance =get_twin_merge_performance(args,models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer,top_k_keep=args.top_k_keep,top_k_remove=args.top_k_remove)

