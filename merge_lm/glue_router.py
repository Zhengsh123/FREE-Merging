from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch
from sklearn.metrics import accuracy_score
from twin_utils import *
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
num_labels = len(glue_tasks)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

def get_ori_datasets(mode="train", max_len=3000):
    data_list = []
    task_name = []
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    for task in glue_data_id_map.keys():
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(
            dataset_name=task, train_split_ratio_for_val=0.1, max_seq_length=128
        )
        if mode == "train":
            data_list.append(train_dataset)
        else:
            data_list.append(test_dataset)
        task_name.append(task)
    max_data_len = max_len
    all_dataset = {}
    data_num = 0
    for idx, i in enumerate(data_list):
        all_dataset[task_name[idx]] = {"input": []}
        for jdx, j in enumerate(i):
            if jdx < max_data_len:
                all_dataset[task_name[idx]]["input"].append(j["input_ids"])
                data_num += 1
            else:
                break
    return all_dataset, data_num

def load_glue(router_info=None):
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    new_datasets = []
    all_dataset = []
    for task in glue_data_id_map.keys():
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(
            dataset_name=task, train_split_ratio_for_val=0.1, max_seq_length=128
        )
        all_dataset.append(test_dataset)
    tot_num = 0
    for sub_data in all_dataset:
        for idx, i in enumerate(sub_data):
            if idx >= 1000:
                continue
            if router_info is not None:
                new_datasets.append({**i, **{"router_prob": router_info[tot_num].tolist()}})
                tot_num += 1
            else:
                new_datasets.append({**i})
    return new_datasets

def generate_router_datasets(
    mode, 
    max_len,
    shared_expert,
):
    datasets, data_num = get_ori_datasets(mode=mode, max_len=max_len)
    ans = {}
    for data_id, (category_name, data_class) in enumerate(datasets.items()):
        data_item = data_class["input"]
        res = []
        for index, data_input in tqdm(enumerate(data_item)):
            res.append((
                index,
                torch.mean(
                    model.forward(
                        input_ids=torch.tensor([data_input]).to(device),
                        output_hidden_states=True
                    )["hidden_states"][-1][0, :, :],
                    dim=0
                ).detach().cpu().numpy(),
            ))    
        # flatten 
        # global_res = sorted([rr for r in res for rr in r], key=lambda x: x[0])
        ans[category_name] = [r[1] for r in res]

    np.savez(f'data/router_{mode}.npz', **ans)

class RouterDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return {
            'input': img,
            'label': target,
        }
    
def load_dataset():

    train_data = np.load(f'data/router_train.npz',allow_pickle=True)
    train_dataset = RouterDataset(
        data = [
            v
            for k in train_data.files
            for v in train_data[k]
        ],
        targets = [
            glue_data_id_map[k] 
            for k in train_data.files
            for _ in range(len(train_data[k]))
        ]
    )
    test_data = np.load(f'data/router_test.npz')
    test_dataset = RouterDataset(
        data = [
            v
            for k in test_data.files
            for v in test_data[k]
        ],
        targets = [
            glue_data_id_map[k] 
            for k in test_data.files
            for _ in range(len(test_data[k]))
        ]
    )
    return {
        'train': train_dataset,
        'test': test_dataset,
    }


def train_router(
    in_domain = None,  # dict
    embed_dims = 768,
):
    encoded_dataset = load_dataset()    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits, labels = torch.tensor(logits), torch.tensor(labels)
        predictions = torch.argmax((logits), dim=-1)
        total = len(labels)
        correct_list = [0] * 8
        total_list = [0] * 8

        # total acc
        correct = predictions.eq((labels)).sum().item()
        acc = correct / total * 100.0
        print(
            "@@ Final {}/{}, Accuracy: {:.2f}%".format(
                correct, total, acc
            )
        )
        # acc per class
        for i in range(8):
            correct_list[i] = ((labels == i) & (predictions == i)).sum().item()
            total_list[i] = (labels == i).sum().item()
        acc_prop = [correct_list[i] / total_list[i] * 100.0 if total_list[i] > 0 else 0 for i in range(8)]
        print("Correct list: ", correct_list)
        print("Accuracy proportion: ", acc_prop)
        return {
            "accuracy": correct / total,
            "accuracy_per_class": acc_prop
        }
    
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir="./data/router",
            evaluation_strategy="epoch",
            save_strategy='epoch',
            learning_rate=0.0005,
            per_device_train_batch_size=1024,
            per_device_eval_batch_size=1024,
            num_train_epochs=50,
            # weight_decay=1e-4,
            logging_steps=20,
            save_total_limit=1,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            ddp_find_unused_parameters=False 
        ),
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./data/router")
    prediction = trainer.predict(encoded_dataset["test"], metric_key_prefix='')

    new_datasets = load_glue(
        tokenizer = AutoTokenizer.from_pretrained('roberta-base'), 
        router_info=prediction.predictions
    )
    json.dump(new_datasets, open('data/test_router.json','w'), ensure_ascii=False)

def main(
    *,
    shared_expert: str = None,
    seed: int = 0,
    train: bool = True,
):
    fix_seed(seed)

    if not os.path.exists('data/test.json') and os.getenv('LOCAL_RANK', 0) == 0:
        data = load_glue(tokenizer = AutoTokenizer.from_pretrained('roberta-base'))
        json.dump(data, open('data/test.json', 'w'))

    # use torchrun to start
    if not os.path.exists('data/router_train.npz'):
        generate_router_datasets('train', 5500, shared_expert)
        generate_router_datasets('test', 1000, shared_expert)
    
    # if not os.path.exists('data/test1.json') or not os.path.exists('outs/router1'):
    if train:
        train_router()
        print('Train Done')

if __name__ == '__main__':
    import defopt
    defopt.run(main)