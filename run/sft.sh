python src/sft.py \
--seed=42 \
--model_name_or_path="Qwen/Qwen2-1.5B" \
--dataset_name='iamnguyen/athena-ultrachat' \
--dataset_train_split='train_sft' \
--dataset_test_split='test_sft' \
--dataset_num_proc=4 \
--report_to=none \
--num_train_epochs=1 \
--max_seq_length=2048 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=128 \
--learning_rate=2e-4 \
--lr_scheduler_type='cosine' \
--optim='paged_adamw_8bit' \
--warmup_ratio=0.01 \
--per_device_eval_batch_size=2 \
--output_dir="output" \
--logging_strategy='steps' \
--logging_steps=1 \
--max_steps=-1 \
--save_strategy='steps' \
--save_total_limit=1 \
--save_steps=4 \
--push_to_hub false \
--hub_strategy='checkpoint' \
--hub_model_id='iamnguyen/ares' \
--gradient_checkpointing \
--resume_from_checkpoint="output/last-checkpoint" 

 

# --use_peft \
# --lora_r=64 \
# --lora_alpha=16 \
# --lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \