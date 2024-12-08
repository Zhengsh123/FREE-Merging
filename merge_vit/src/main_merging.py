import torch
import argparse
from eval import *
from utils import *
from task_vectors import TaskVectorFFT,TaskVectorMiddleKeep,TaskVectorABC,TiesMerge

parser = argparse.ArgumentParser(description="new parser")
parser.add_argument('--config_root_path',type=str,default='./config')
parser.add_argument('--model',type=str,default='ViT-B-32')
parser.add_argument('--task',type=str,default='8')
parser.add_argument('--method',type=str,default='FR')
parser.add_argument("--device",type=str,default='cuda:1')
args = parser.parse_args()
args=read_config(args)

def merging_8(args):
    logger=create_log_dir(args)
    pretrained_checkpoint=f'{args.model_home}/{args.model}/zeroshot.pt'
    datasets=get_dataset_name(args)
    ## define shared vector
    if args.method=='FR' or args.method=='FREE':
        shared_vectors = [TaskVectorFFT(pretrained_checkpoint, f'{args.model_home}/{args.model}/{dataset}/finetuned.pt',args=args) for dataset in datasets]
    elif args.method=='bread':
        shared_vectors = [TaskVectorMiddleKeep(pretrained_checkpoint, f'{args.model_home}/{args.model}/{dataset}/finetuned.pt',top_k_keep=args.special.top_k_keep,top_k_remove=args.special.top_k_remove) for dataset in datasets]
    elif args.method=='simple':
        shared_vectors = [TaskVectorABC(pretrained_checkpoint, f'{args.model_home}/{args.model}/{dataset}/finetuned.pt') for dataset in datasets]
    if args.special.with_align:
        shared_vectors=align_each_layer(shared_vectors)
    # shared_task_vector_sum = TiesMerge(pretrained_checkpoint, list_finetuned_checkpoints=shared_vectors,top_k_keep=0.2)
    # shared_task_vector_sum = sum(shared_vectors)
    if args.special.shared_only==True: # without expert
        res={}
        shared_task_vector_sum = sum(shared_vectors)
        for dataset in datasets:
            image_encoder = shared_task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=args.special.scaling_coef)
            metrics=eval_single_dataset_30(image_encoder, dataset, args)
            res[dataset]=round(metrics['top1'],4)
        log_training_results(logger,args.special,res)
    else: # with expert
        shared_task_vector_sum = sum(shared_vectors)
        task_experts=[TaskVectorMiddleKeep(pretrained_checkpoint, f'{args.model_home}/{args.model}/{dataset}/finetuned.pt',top_k_keep=args.special.top_k_keep,top_k_remove=args.special.top_k_remove) for dataset in datasets]
        rescaling=cal_rescaling(shared_vectors,task_experts,args.special.top_k_keep)
        task_experts_dict={}
        for i,data in enumerate(datasets):
            task_expert=task_experts[i]*rescaling[i]
            task_experts_dict[data]=task_expert
        router=torch.load(args.special.router_path)
        router=router.to(args.device)
        router.eval()
        res={}
        for i,dataset in enumerate(datasets):
            metrics=eval_single_dataset_with_router(datasets, dataset, args,pretrained_checkpoint,shared_vector=shared_task_vector_sum,task_expert_dict=task_experts_dict,router=router)
            res[dataset]=round(metrics['top1'],4)
        log_training_results(logger,args.special,res)
        

if __name__=="__main__":
    merging_8(args)