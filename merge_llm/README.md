# Merging Large Language Models

### Dependencies

Please follow [della](https://github.com/declare-lab/della) and  [MergeKit](https://github.com/arcee-ai/mergekit) to install dependencies and download datasets.

### Run

Before performing pruning or merging, add the paths to the following model checkpoints in `merge.py`

```
# Expert model paths
WIZARDMATH13B_PATH = "<Path to WizardMath-13B-V1.0>"
WIZARDCODER13B_PATH = "<Path to WizardCoder-Python-13B-V1.0>"
WIZARDLM13B_PATH = "<Path to WizardLM-13B-V1.2>"
LLAMA2_13B_CODE_ALPACA = "<Path to llama-2-13b-code-alpaca>"

# Base model paths
CODELLAMA_PATH = "<Path to CodeLlama-13b-Python-hf>"
LLAMA2_13B_PATH = "<Path to Llama-2-13b-hf>"
```

You can easily run our FR-Merging and FREE-Merging using the following commands. 

```
# For FR-Merging
python merge.py --density 0.9 --gamma 0.01 --merge_method fourier --models LM_math_code --weights 0.8 --lambda_factor 1.1 --window_size 0.14 --rescale 1 --seed 42 --drop_rate 0.3
# For FREE-Merging
python merge.py --density 0.9 --gamma 0.01 --merge_method FREE --models LM_math_code --weights 0.8 --lambda_factor 1.1 --window_size 0.14 --rescale 1 --seed 42 --drop_rate 0.3 --task_name [task_name]
```

### Eval

Please follow [della](https://github.com/declare-lab/della) for details of evaluation of merging results. We use [HumanEval](https://github.com/openai/human-eval) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation of AlpacaEval, MBPP, and GSM8K.

## Acknowledgement

Our implementation references the code below, thanks to them.

Task-Vectors: [task_vectors: Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors)

EMR-Merging: [EMR-Merging: Tuning-Free High-Performance Model Merging](https://github.com/harveyhuang18/EMR_Merging)

Twin-Merging: [Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://github.com/LZY-the-boys/Twin-Merging)

Ties-Merging: https://github.com/prateeky2806/ties-merging/tree/main

MergeKit: [arcee-ai/mergekit: Tools for merging pretrained large language models.](https://github.com/arcee-ai/mergekit)

BEiT-3: https://github.com/microsoft/unilm/tree/master/beit3



