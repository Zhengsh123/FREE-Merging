# Merging Language Models

### Dependencies

Please follow [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging) and [Twin-Merging](https://github.com/LZY-the-boys/Twin-Merging) to install the dependencies.

### Checkpoints

ou can download the fine-tuned checkpoints from huggingface [here](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts/tree/main) or [here]([lu-vae/roberta-glue · Hugging Face](https://huggingface.co/lu-vae/roberta-glue)). You can refer to [Twin-Merging](https://github.com/LZY-the-boys/Twin-Merging) and [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging) to download and place the models accordingly.

### Datasets

You can modify the `cache_dir` in the `utils_file/load_config.py` file to specify your own path to save the datasets.

### Run

You can easily run our FR-Merging and FREE-Merging using the following commands. The default parameters can be modified in the `config` folder.

```
# For FR-Merging
python merge_roberta_glue.py --merging_method_name fr_merging
# For FREE-Merging
python merge_roberta_glue.py --merging_method_name free_merging
# Another way for FREE-Merging 
bash run.sh
```

## Acknowledgement

Our implementation references the code below, thanks to them.

Task-Vectors: [task_vectors: Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors)

EMR-Merging: [EMR-Merging: Tuning-Free High-Performance Model Merging](https://github.com/harveyhuang18/EMR_Merging)

Twin-Merging: [Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://github.com/LZY-the-boys/Twin-Merging)

Ties-Merging: https://github.com/prateeky2806/ties-merging/tree/main

MergeKit: [arcee-ai/mergekit: Tools for merging pretrained large language models.](https://github.com/arcee-ai/mergekit)

BEiT-3: https://github.com/microsoft/unilm/tree/master/beit3



