#!/bin/bash

echo "=== 开始使用 Jittor 进行 E2E 数据集微调 ==="

# 步骤 1: 使用 LoRA 训练 GPT - 2 模型
echo "【步骤 1/4】正在训练模型 (Jittor)..."
python src/gpt2_ft.py \
--log_dir ./log/e2e \
--train_data ./data/e2e/train.jsonl \
--valid_data ./data/e2e/valid.jsonl \
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
--work_dir ./trained_models/GPT2_SM/e2e \
--random_seed 100

# 步骤 2: 使用 beam search 生成输出
echo "【步骤 2/4】正在生成预测结果 (Jittor)..."
python src/gpt2_beam.py \
--data ./data/e2e/test.jsonl \
--log_dir ./log/e2e \
--batch_size 1 \
--seq_len 256 \
--eval_len 64 \
--model_card gpt2.sm \
--init_checkpoint ./trained_models/GPT2_SM/e2e/model.3000.pt \
--platform local \
--lora_dim 4 \
--lora_alpha 32 \
--beam 10 \
--length_penalty 0.8 \
--no_repeat_ngram_size 4 \
--repetition_penalty 1.0 \
--eos_token_id 628 \
--work_dir ./trained_models/GPT2_SM/e2e \
--output_file predict.3000.b10p08r4.jsonl

# 步骤 3: 解码生成的输出
echo "【步骤 3/4】正在解码预测结果..."
python src/gpt2_decode.py \
--log_dir ./log/e2e \
--vocab ./vocab \
--sample_file ./trained_models/GPT2_SM/e2e/predict.3000.b10p08r4.jsonl \
--input_file ./data/e2e/test_formatted.jsonl \
--output_ref_file e2e_ref.txt \
--output_pred_file e2e_pred.txt

# 步骤 4: 在 E2E 测试集上运行评估
echo "【步骤 4/4】正在计算评估指标..."
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p

echo "=== 已完成 ==="