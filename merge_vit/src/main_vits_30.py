import torch
import imageio
import argparse
from eval import *
from utils import *
import torch.optim.optimizer
from transformers.models.vit.modeling_vit import *
from PIL import Image
from transformers import ViTForImageClassification
from tqdm import tqdm
import copy

from torch.utils.data.dataloader import DataLoader
from task_vectors import TaskVectorFFT_Init_Tensor,TaskVectorBREAD_Init_Tensor

parser = argparse.ArgumentParser(description="new parser")
parser.add_argument('--config_root_path',type=str,default='./config')
parser.add_argument('--model',type=str,default='ViT-B-16')
parser.add_argument('--task',type=str,default='30')
parser.add_argument('--method',type=str,default='FR')
parser.add_argument("--device",type=str,default='cuda:1')
args = parser.parse_args()
args=read_config(args)

def eval_single_dataset_30(image_encoder, dataset, clf, bz, device, ds_name='None'):
    image_encoder.classifier = clf
    model = image_encoder.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=bz)
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm(dataloader)):
            x = data[0].to(device)
            y = data[1].to(device)
            logits = model(x).logits
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {ds_name}. Accuracy: {100 * top1:.2f}%')
    return metrics

def eval_single_dataset_with_router_30(dataset, dataset_name_list,cur_dataset_name,args,image_encoder,classifiers,shared_vector,task_expert_dict,router):
    model_dict={}
    pretrained_state_dict=image_encoder.vit.state_dict()
    for key in task_expert_dict.keys():
        cur_image_encoder=copy.deepcopy(image_encoder)
        clf=classifiers[key]
        task_expert=task_expert_dict[key]
        cur_image_encoder.classifier = clf
        task_vector=task_expert.sum_two(shared_vector*0.1).vector
        task_vector=apply_to_pretrain(pretrained_state_dict,task_vector,args)
        cur_image_encoder.vit.load_state_dict(task_vector, strict=False)
        cur_image_encoder.eval()
        model_dict[key]=cur_image_encoder
    dataloader = DataLoader(dataset, batch_size=args.batch_size,num_workers=args.num_workers)
    device = args.device
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            outputs=router(x)
            _, predicted = torch.max(outputs, 1)
            preds=[dataset_name_list[idx] for idx in predicted.cpu()]
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
                model=model.to(device)
                model_output = model(model_inputs).logits
                for idx, output in zip(grouped_indices[label], model_output):
                    pred[idx] = output.argmax(dim=0, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {cur_dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics

def apply_to_pretrain(pretrained_state_dict,task_vector_state_dict,args):
    task_vector={}
    with torch.no_grad():
        for key in pretrained_state_dict:
            if key not in task_vector_state_dict:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            if 'attention' in key or 'output' in key or 'intermediate' in key:
                task_vector[key] = pretrained_state_dict[key] + task_vector_state_dict[key]*args.special.scaling_coef
            else:
                task_vector[key] = pretrained_state_dict[key]
    return task_vector

def merging_30(args):
    logger=create_log_dir(args)
    pretrained_checkpoint = 'google/vit-base-patch16-224-in21k'
    test_datasets=get_30_test_datasets()
    finetune_checkpoints=get_30_finetune()
    ## load model
    pretrained_model = ViTForImageClassification.from_pretrained(pretrained_checkpoint).to('cpu')
    pretrained_state_dict=pretrained_model.vit.state_dict()
    finetune_state_dicts,classifiers={},{}
    for key in tqdm(finetune_checkpoints, 'loading_model_weights'):
        ft_check = ViTForImageClassification.from_pretrained(finetune_checkpoints[key]).to('cpu').vit.state_dict()
        finetune_state_dicts[key]=ft_check
        classifiers[key]=ViTForImageClassification.from_pretrained(finetune_checkpoints[key]).to('cpu').classifier
    if args.method=='FR' or args.method=='FREE':
        shared_vectors=[TaskVectorFFT_Init_Tensor(pretrained_checkpoint=pretrained_state_dict,finetuned_checkpoint=finetune_state_dicts[key],args=args) for key in finetune_state_dicts]
    elif args.method=='bread':
        shared_vectors=[TaskVectorBREAD_Init_Tensor(pretrained_checkpoint=pretrained_state_dict,finetuned_checkpoint=finetune_state_dicts[key],top_k_keep=args.special.top_k_keep,top_k_remove=args.special.top_k_remove) for key in finetune_state_dicts]
    elif args.metethod=='simple':
        shared_vectors=[TaskVectorBREAD_Init_Tensor(pretrained_checkpoint=pretrained_state_dict,finetuned_checkpoint=finetune_state_dicts[key],top_k_keep=1.0,top_k_remove=0.0) for key in finetune_state_dicts]
    if args.special.with_align:
        shared_vectors=align_each_layer(shared_vectors)
    if args.special.shared_only==True: # without expert
        res={}
        shared_task_vector_sum = sum(shared_vectors)
        for i,key in enumerate(test_datasets):
            dataset=test_datasets[key]
            image_encoder = ViTForImageClassification.from_pretrained(pretrained_checkpoint)
            pretrained_state_dict=image_encoder.vit.state_dict()
            task_vector=shared_task_vector_sum.vector
            task_vector=apply_to_pretrain(pretrained_state_dict,task_vector,args)
            image_encoder.vit.load_state_dict(task_vector, strict=False)
            metrics = eval_single_dataset_30(image_encoder, dataset, classifiers[key], args.batch_size, args.device, key)
            res[key]=round(metrics['top1'],4)
        log_training_results(logger,args.special,res) 
    else:
        res={}
        shared_task_vector_sum = sum(shared_vectors)
        image_encoder = ViTForImageClassification.from_pretrained(pretrained_checkpoint)
        task_experts=[TaskVectorBREAD_Init_Tensor(pretrained_checkpoint=pretrained_state_dict,finetuned_checkpoint=finetune_state_dicts[key],top_k_keep=args.special.top_k_keep,top_k_remove=args.special.top_k_remove) for key in finetune_state_dicts]
        rescaling=cal_rescaling(shared_vectors,task_experts,args.special.top_k_keep)
        task_experts_dict={}
        for i,data in enumerate(test_datasets):
            task_expert=task_experts[i]*7#rescaling[i]
            task_experts_dict[data]=task_expert
        router=torch.load(args.special.router_path)
        router=router.to(args.device)
        router.eval()
        res={}
        for i,key in enumerate(test_datasets):
            metrics=eval_single_dataset_with_router_30(test_datasets[key], list(test_datasets.keys()),key,args,image_encoder,classifiers=classifiers,shared_vector=shared_task_vector_sum,task_expert_dict=task_experts_dict,router=router)
            res[key]=round(metrics['top1'],4)
        log_training_results(logger,args.special,res)

if __name__=="__main__":
    merging_30(args)