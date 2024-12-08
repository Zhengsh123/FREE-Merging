import os
import json
import tqdm

import torch
import numpy as np

import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier

from src.datasets.registry import get_dataset

def eval_single_dataset_30(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset_30(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def eval_single_dataset_with_router(datasets, dataset_name, args,pretrained_checkpoint,shared_vector,task_expert_dict,router):
    model_dict={}
    for key in task_expert_dict.keys():
        classification_head = get_classification_head(args, dataset_name)
        task_expert=task_expert_dict[key]
        location_vector=task_expert.sum_two(shared_vector*args.special.scaling_coef)
        image_encoder=location_vector.apply_to(pretrained_checkpoint, scaling_coef=args.special.scaling_coef)
        model = ImageClassifier(image_encoder, classification_head).to(args.device)
        model.eval()
        model_dict[key]=model
    cur_model=model_dict[dataset_name]
    dataset = get_dataset(
        dataset_name,
        cur_model.val_preprocess,
        location=args.data_location if dataset_name!='ImageNet' else args.data_location+'/imagenet/',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            outputs=router(x)
            _, predicted = torch.max(outputs, 1)
            preds=[datasets[idx] for idx in predicted.cpu()]
            grouped_inputs = {label: [] for label in model_dict.keys()}
            grouped_indices = {label: [] for label in model_dict.keys()}
            for j, pred_label in enumerate(preds):
                grouped_inputs[pred_label].append(x[j])
                grouped_indices[pred_label].append(j) 
            pred = torch.zeros(len(x)).to(device)
            for label, model in model_dict.items():
                if len(grouped_inputs[label]) == 0:
                    continue 
                model_inputs = torch.stack(grouped_inputs[label]).to(device)
                model_output = model(model_inputs)
                for idx, output in zip(grouped_indices[label], model_output):
                    pred[idx] = output.argmax(dim=0, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics
