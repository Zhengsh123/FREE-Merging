import os
import yaml
import time
import logging
import pickle
import torch
import argparse
import numpy as np
import torch.nn as nn
from argparse import Namespace
from torchvision import models
from torchvision.transforms import (Compose,Normalize,Resize,Grayscale,ToTensor)
from torchvision.datasets import MNIST, ImageFolder, CIFAR10, CIFAR100, EuroSAT, Food101, GTSRB, SVHN, FashionMNIST, \
    OxfordIIITPet, FGVCAircraft, FER2013, STL10, EMNIST, DTD, SUN397, StanfordCars
from src.datasets.resisc45 import RESISC45Dataset

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def safe_load_state_dict(weights_path: str):
    try:
        state_dict = torch.load(weights_path).state_dict()
    except:
        with open(weights_path, "rb") as f:
            obj = f.read()
        state_dict = pickle.loads(obj, encoding="latin1").state_dict()
    return state_dict

def read_config(args):
    yml_path="{}/{}_{}_{}.yaml".format(args.config_root_path,args.model,args.task,args.method)
    with open(yml_path, 'r') as f:
        default_arg = yaml.safe_load(f)
    args = dict2namespace({**vars(args), **default_arg})
    return args

def create_log_dir(args):
    if not os.path.exists(args.log_root_path):
        os.makedirs(args.log_root_path)
    logger = logging.getLogger(args.log_root_path)
    logger.setLevel(logging.DEBUG)
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    filename="{}_{}_{}_{}_{}.txt".format(args.model,args.task,args.method,str(args.special.with_align),str_time_)
    fh = logging.FileHandler(args.log_root_path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def log_training_results(logger, params: Namespace, results: dict):
    logger.info("===== Training Parameters =====")
    for key, value in vars(params).items():
        logger.info(f"{key}: {value}")
    logger.info("===== Results =====")
    for dataset, performance in results.items():
        logger.info(f"Dataset: {dataset}: {performance}")
    logger.info("\n")

def get_dataset_name(args):
    if args.task=='8':
        return ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SVHN', 'SUN397']
        # return ['SUN397','Cars','RESISC45','EuroSAT','SVHN','GTSRB','MNIST','DTD']
    elif args.task=='30':
        pass

def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


def collect_each_dataset_layer_mean(task_vectors,datasets=None):
    dataset_layer_mean=[]
    for task_vector in task_vectors:
        cur_layer_mean=[]
        cur_vector=task_vector.vector
        for key in cur_vector.keys():
            cur_layer_mean.append(torch.mean(torch.abs(cur_vector[key])).cpu().item())
        dataset_layer_mean.append(cur_layer_mean)
    
    return dataset_layer_mean

def align_each_layer(task_vectors):
    dataset_layer_mean=collect_each_dataset_layer_mean(task_vectors)
    array_dataset_layer_mean = np.array(dataset_layer_mean)
    col_max = np.max(array_dataset_layer_mean, axis=0)
    result = 1/(array_dataset_layer_mean / col_max)
    result= result.tolist()
    new_task_vectors=[]
    for task_vector,res in zip(task_vectors,result):
        for key,ratio in zip(task_vector.vector.keys(),res):
            task_vector.vector[key]*=ratio
        new_task_vectors.append(task_vector)
    return new_task_vectors

def cal_rescaling(task_vectors,expert_vectors,keep_ratio):
    dataset_layer_mean=collect_each_dataset_layer_mean(task_vectors)
    array_dataset_layer_mean = np.array(dataset_layer_mean)
    row_means = np.mean(array_dataset_layer_mean, axis=1)
    mean_sum = np.sum(row_means)
    normalized_array = row_means / mean_sum

    expert_layer_mean=collect_each_dataset_layer_mean(expert_vectors)
    array_expert_layer_mean = np.array(expert_layer_mean)
    expert_row_means = np.mean(array_expert_layer_mean, axis=1)/keep_ratio
    rescaling=(expert_row_means*np.log10(keep_ratio))/(-normalized_array*mean_sum)
    return rescaling

class ResNet18WithMLP(nn.Module):
    def __init__(self, num_datasets, feature_dim=512):
        super(ResNet18WithMLP, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_datasets)
        )
    def forward(self, x):
        features = self.resnet18(x)
        features = features.view(features.size(0), -1)
        out = self.mlp(features)
        return out
    
class SimpleMLP(nn.Module):
    def __init__(self, num_datasets, hidden_units=[512, 256]):
        super(SimpleMLP, self).__init__()
        layers = []
        current_size = 224*224*3
        for units in hidden_units:
            layers.append(nn.Linear(current_size, units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
            current_size = units
        layers.append(nn.Linear(current_size, num_datasets))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def _convert_to_rgb(image):
    return image.convert('RGB')

trans_gray = Compose([
    Resize(size=(224, 224)),
    Grayscale(3),
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])
trans_rgb = Compose([
    Resize(size=(224, 224)),
    _convert_to_rgb,
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

def get_30_test_datasets():
    datasets={
       'MNIST': MNIST(root="./dataset/",train=False,transform=trans_gray),
       'CIFAR10': CIFAR10(root="./dataset/",train=False,transform=trans_rgb),
       'Vegetables': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Food-101': Food101(root="./dataset/",split='test',transform=trans_rgb),
       'Kvasir-V2': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Intel-Images': ImageFolder(root="./dataset/",transform=trans_rgb),

       'Cars': StanfordCars(root="./dataset/",split='test',transform=trans_rgb),
       'EuroSAT': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Weather': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Cats & Dogs': ImageFolder(root="./dataset/",transform=trans_rgb),
       'MangoLeafBD': ImageFolder(root="./dataset/",transform=trans_rgb),
       'beans': ImageFolder(root="./dataset/",transform=trans_rgb),

       'CIFAR100': CIFAR100(root="./dataset/",train=False,transform=trans_rgb),
       'GTSRB': GTSRB(root="./dataset/",split='test',transform=trans_rgb),
       'SVHN': SVHN(root="./dataset/",split='test',transform=trans_rgb),
       'Dogs': ImageFolder(root="./dataset/",transform=trans_rgb),
       'FashionMNIST': FashionMNIST(root="./dataset/",train=False,transform=trans_gray),
       'OxfordIIITPet': OxfordIIITPet(root="./dataset/",split='test',transform=trans_rgb),

       'Landscape': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Flowers': ImageFolder(root="./dataset/",transform=trans_rgb),
       'STL10': STL10(root="./dataset/",split='test',transform=trans_rgb),
       'CUB-200-2011': ImageFolder(root="./dataset/",transform=trans_rgb),
       'EMNIST': EMNIST(root="./dataset/",split='letters',train=False,transform=trans_gray),
       'DTD': ImageFolder(root="./dataset/",transform=trans_rgb),

       'RESISC45': RESISC45Dataset(root="./dataset/",split='test',transforms=trans_rgb),
       'SUN397': ImageFolder(root="./dataset/",transform=trans_rgb),
       'KenyanFood13': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Animal-10N': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Garbage': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Fruits-360': ImageFolder(root="./dataset/",transform=trans_rgb),
    }
    return datasets

def get_30_train_datasets():
    datasets={
       'MNIST': MNIST(root="./dataset/",train=True,transform=trans_gray),
       'CIFAR10': CIFAR10(root="./dataset/",train=True,transform=trans_rgb),
       'Vegetables': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Food-101': Food101(root="./dataset/",split='train',transform=trans_rgb),
       'Kvasir-V2': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Intel-Images': ImageFolder(root="./dataset/",transform=trans_rgb),

       'Cars': StanfordCars(root="./dataset/",split='train',transform=trans_rgb),
       'EuroSAT': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Weather': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Cats & Dogs': ImageFolder(root="./dataset/",transform=trans_rgb),
       'MangoLeafBD': ImageFolder(root="./dataset/",transform=trans_rgb),
       'beans': ImageFolder(root="./dataset/",transform=trans_rgb),

       'CIFAR100': CIFAR100(root="./dataset/",train=True,transform=trans_rgb),
       'GTSRB': GTSRB(root="./dataset/",split='train',transform=trans_rgb),
       'SVHN': SVHN(root="./dataset/",split='train',transform=trans_rgb),
       'Dogs': ImageFolder(root="./dataset/",transform=trans_rgb),
       'FashionMNIST': FashionMNIST(root="./dataset/",train=False,transform=trans_gray),
       'OxfordIIITPet': OxfordIIITPet(root="./dataset/",split='trainval',transform=trans_rgb),

       'Landscape': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Flowers': ImageFolder(root="./dataset/",transform=trans_rgb),
       'STL10': STL10(root="./dataset/",split='train',transform=trans_rgb),
       'CUB-200-2011': ImageFolder(root="./dataset/",transform=trans_rgb),
       'EMNIST': EMNIST(root="./dataset/",split='letters',train=True,transform=trans_gray),
       'DTD': ImageFolder(root="./dataset/",transform=trans_rgb),

       'RESISC45': RESISC45Dataset(root="./dataset/",split='train',transforms=trans_rgb),
       'SUN397': ImageFolder(root="./dataset/",transform=trans_rgb),
       'KenyanFood13': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Animal-10N': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Garbage': ImageFolder(root="./dataset/",transform=trans_rgb),
       'Fruits-360': ImageFolder(root="./dataset/",transform=trans_rgb),
    }
    return datasets

def get_30_finetune():
    ckpts={
       'MNIST':"./model/ViT-B-16/mnist_finetuned/",
       'CIFAR10': "./model/ViT-B-16/cifar10_finetuned/",
       'Vegetables': "./model/ViT-B-16/vegetables_finetuned/",
       'Food-101': "./model/ViT-B-16/food101_finetuned/",
       'Kvasir-V2': "./model/ViT-B-16/kvasirv2_finetuned/",
       'Intel-Images': "./model/ViT-B-16/intel_finetuned/",

       'Cars': "./model/ViT-B-16/stanford_cars_finetuned_mine/",
       'EuroSAT': "./model/ViT-B-16/eurosat_finetuned/",
       'Weather': "./model/ViT-B-16/weather_finetuned/",
       'Cats & Dogs': "./model/ViT-B-16/cats_vs_dogs_finetuned/",
       'MangoLeafBD': "./model/ViT-B-16/mango_leaf_finetuned/",
       'beans': "./model/ViT-B-16/beans_finetuned/",

       'CIFAR100': "./model/ViT-B-16/cifar100_finetuned/",
       'GTSRB': "./model/ViT-B-16/gtsrb_finetuned/",
       'SVHN': "./model/ViT-B-16/svhn_finetuned/",
       'Dogs': "./model/ViT-B-16/dogs_finetuned/",
       'FashionMNIST': "./model/ViT-B-16/fashion_finetuned/",
       'OxfordIIITPet': "./model/ViT-B-16/pets_finetuned/",

       'Landscape': "./model/ViT-B-16/landscape_finetuned/",
       'Flowers': "./model/ViT-B-16/flowers_finetuned/",
       'STL10': "./model/ViT-B-16/stl10_finetuned/",
       'CUB-200-2011': "./model/ViT-B-16/cub200_finetuned/",
       'EMNIST': "./model/ViT-B-16/emnist_finetuned/",
       'DTD': "./model/ViT-B-16/dtd_finetuned/",

       'RESISC45': "./model/ViT-B-16/resisc45_finetuned/",
       'SUN397': "./model/ViT-B-16/sun397_finetuned/",
       'KenyanFood13': "./model/ViT-B-16/food13_finetuned/",
       'Animal-10N': "./model/ViT-B-16/animal_10n_finetuned_mine/",
       'Garbage': "./model/ViT-B-16/garbage_classification_finetuned/",
       'Fruits-360': "./model/ViT-B-16/fruits_finetuned/",
    }
    return ckpts