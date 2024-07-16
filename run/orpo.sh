python src/orpo.py \
--seed=42 \
--model_name_or_path="Qwen/Qwen2-0.5B" \
--load_in_4bit \
--bnb_4bit_quant_type=nf4 \
--use_bnb_nested_quant=false \
--use_peft \
--lora_r=128 \
--lora_alpha=256 \
--lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
--dataset='iamnguyen/orpo-dpo-mix-translated' \
--dataset_num_proc=4 \
--report_to='none' \
--num_train_epochs=1 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--learning_rate=5e-5 \
--lr_scheduler_type='cosine' \
--optim='paged_adamw_8bit' \
--warmup_ratio=0.01 \
--per_device_eval_batch_size=2 \
--output_dir="output" \
--logging_strategy='steps' \
--logging_steps=1 \
--max_steps=1 \
--save_strategy='steps' \
--save_total_limit=1 \
--save_steps=4 \
--push_to_hub false \
--hub_strategy='checkpoint' \
--push_to_hub_model_id='iamnguyen/whale' \
--gradient_checkpointing \
--resume_from_checkpoint="output/last-checkpoint" 



