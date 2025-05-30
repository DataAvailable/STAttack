In this paper, we measure and interpret the impacts of code synonym transformation on the performance of deep vulnerability detection models, thereby providing guidance for optimizing the vulnerability detection capabilities. Our work mainly contains three parts: 1) We generalize eight synonymous transformation rules for constructing code synonymous adversarial examples, which have higher complexity and diversity with semantic preservation relative to the adversarial examples generated in previous work, to better measure the robustness of vulnerability detection models. Experiments on publicly available datasets show that these vulnerability detection models drop 15.67%-83.84% and 9.95%-86.26% in Recall and F1 values, respectively. 2) We explain the impact of code synonymous adversarial examples on the performance of vulnerability detection models in terms of two phases: code representations (input) and feature extraction (model). 3) We adopt synonym adversarial training to enhance the robustness of vulnerability detection and improve detection performance, achieving maximum improvements of 86.23% and 19.77% respectively compared to the original model.

![](https://github.com/DataAvailable/STAttack/blob/main/Figures/Figure1.png?raw=true){:height="50%" width="50%"}

## Requirements
```
pandas=2.0.3
scikit-learn=1.3.2
scipy=1.10.1
seaborn=0.13.2
sh=2.2.2
shap=0.44.1
tensorboardx=2.6.2.2
tokenizers=0.13.3
tqdm=4.67.1
transformers=4.30.0
umap=0.1.1
xz=5.6.4
urllib3=2.2.3
```
## Datasets
[1] Big-Vul: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/blob/master/all_c_cpp_release2.0.csv

[2] FFmpeg+Qemu(or Devign): https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit

## Target Models
[1] LineVul: https://github.com/awsm-research/LineVul

[2] Devign: https://github.com/epicosy/devign

[3] ReGVD: https://github.com/daiquocnguyen/GNN-ReGVD

## How to Reproduce？

------------------------------------------------------------
1.1 Download the dataset.
```
cd LineVul/data
cd big-vul_dataset
wget https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
wget https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
wget https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
```

1.2 Download the original linevul model.
```
cd ..
cd LineVul/linevul
cd saved_models/checkpoint-best-f1
wget https://drive.google.com/uc?id=1oodyQqRb9jEcvLMVVKILmu8qHyNwd-zH
```

1.3 To reproduce the original detection performance, run the following commands (original trained model, original testing dataset):
```
cd LineVul/linevul
python linevul_Original.py \
    --model_name=12heads_linevul_model.bin \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --test_data_file=../data/big-vul_dataset/test.csv \
    --block_size 512 \
    --eval_batch_size 512
```

1.4 To reproduce the detection performance after Synonymous Transformation attack, run the following commands (original trained model, Transformed testing dataset):
```
cd LineVul/linevul
python linevul_STAttack.py \
    --model_name=12heads_linevul_model.bin \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --test_data_file=../data/big-vul_dataset/test.csv \
    --block_size 512 \
    --eval_batch_size 512
    --transformation_rules R1
    --filter_error yes
```
The value of the parameter `transformation_rules` is `R1-R8`.

------------------------------------------------------------
2.1 Preprocess the dataset.
```
cd ReGVD/dataset
wget https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
python preprocess.py
cd ..
```

2.2 To reproduce the original detection performance, run the following commands:
```
cd code
python run_Original.py 
	--output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 \
	--model_type=roberta \
	--tokenizer_name=microsoft/graphcodebert-base \
	--model_name_or_path=microsoft/graphcodebert-base \
	--do_train \
	--do_eval \
	--do_test \
	--train_data_file=../dataset/train.jsonl \
	--eval_data_file=../dataset/valid.jsonl \
	--test_data_file=../dataset/test.jsonl \
	--block_size 400 \
	--train_batch_size 128 \
	--eval_batch_size 128 \
	--max_grad_norm 1.0 \
	--evaluate_during_training \
	--gnn ReGCN \
	--learning_rate 5e-4 \
	--epoch 100 \
	--hidden_size 128 \
	--num_GNN_layers 2 \
	--format uni \
	--window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt
```

2.3 To reproduce the detection performance after Synonymous Transformation attack, run the following commands (original trained model, Transformed testing dataset):
```
cd code
python run_STAttack.py
	--output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 \
	--model_type=roberta \
	--tokenizer_name=microsoft/graphcodebert-base \
	--model_name_or_path=microsoft/graphcodebert-base \
	--do_test \
	--test_data_file=../dataset/test.jsonl \
	--block_size 400 \
	--max_grad_norm 1.0 \
	--evaluate_during_training \
	--gnn ReGCN \
	--hidden_size 128 \
	--num_GNN_layers 2 \
	--format uni \
	--window_size 5 \
	--transformation_rules R1
	--filter_error yes
	--seed 123456 2>&1 | tee test.log
```








