#!/bin/bash

echo "=== 开始使用 PyTorch 进行 DART 数据集微调 ==="

# 步骤 1: 使用 LoRA 训练 GPT-2 模型
echo "【步骤 1/4】正在训练模型 (PyTorch)..."
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
--log_dir ./log/dart \
--train_data ./data/dart/train.jsonl \
--valid_data ./data/dart/valid.jsonl \
--train_batch_size 4 \
--grad_acc 1 \
--valid_batch_size 1 \
--seq_len 256 \
--model_card gpt2.sm \
--init_checkpoint ./pretrained_checkpoints/gpt2-pytorch_model.bin \
--platform local \
--clip 0.0 \
--lr 0.0002 \
--weight_decay 0.01 \
--correct_bias \
--adam_beta2 0.999 \
--scheduler linear \
--warmup_step 500 \
--max_epoch 3 \
--save_interval 1000 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--label_smooth 0.1 \
--work_dir ./trained_models/GPT2_SM/dart \
--random_seed 100

# 步骤 2: 使用 beam search 生成输出
echo "【步骤 2/4】正在生成预测结果 (PyTorch)..."
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
--data ./data/dart/test.jsonl \
--log_dir ./log/dart \
--batch_size 1 \
--seq_len 256 \
--eval_len 64 \
--model_card gpt2.sm \
--init_checkpoint ./trained_models/GPT2_SM/dart/model.2250.pt \
--platform local \
--lora_dim 4 \
--lora_alpha 32 \
--beam 10 \
--length_penalty 0.8 \
--no_repeat_ngram_size 4 \
--repetition_penalty 1.0 \
--eos_token_id 628 \
--work_dir ./trained_models/GPT2_SM/dart \
--output_file predict.2250.b10p08.jsonl

# 步骤 3: 解码生成的输出（与 Jittor 通用）
echo "【步骤 3/4】正在解码预测结果..."
python src/gpt2_decode.py \
--log_dir ./log/dart \
--vocab ./vocab \
--sample_file ./trained_models/GPT2_SM/dart/predict.2250.b10p08.jsonl \
--input_file ./data/dart/test_formatted.jsonl \
--ref_type dart \
--ref_num 6 \
--output_ref_file eval/GenerationEval/data/references_dart \
--output_pred_file eval/GenerationEval/data/hypothesis_dart \
--tokenize --lower

# 步骤 4: 在 DART 测试集上运行评估（与 Jittor 通用）
echo "【步骤 4/4】正在计算评估指标..."
cd ./eval/GenerationEval/
python eval.py \
-R data/references_dart/reference \
-H data/hypothesis_dart \
-nr 6 \
-m bleu,meteor,ter
cd ../..

echo "=== 已完成 ==="