import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from itertools import chain
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from trl import (
    TrlParser, 
    SFTScriptArguments, 
    SFTConfig, 
    SFTTrainer,
    ModelConfig, 
    get_quantization_config, 
    get_kbit_device_map, 
    get_peft_config
)


@dataclass
class ExtraConfig:
    '''
    Some extra arguments for pretrain step
    '''
    do_pretraining: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, auto using configs for pretrain mode"
            )
        },
    )
    group_text_num_procs: int = field(default=1, metadata={"help": ("Num procs for group text pretraining")})
    # do_mixing_datasets: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "If true, using dataset mixer for mixing datasets"
    #         )
    #     }
    # )
    

# def mix_datasets(dataset_mixer, shuffle=True, seed=42, test_percentage=0.01):
#     '''
#     Example format for dataset_nmixer:
#     dataset_mixer = {
#             "dataset1": 1, # dataset_name: proportion
#             # "dataset1": 0.3,
#             # "dataset1": 0.2,
#                 }
#     '''
#     COLUMNS_TO_KEEP = ['text']
#     raw_train_datasets = []
#     raw_test_datasets = []
#     new_dataset = DatasetDict()
#     for key, value in dataset_mixer.items():
#         if key == "HuggingFaceTB/cosmopedia":
#             dataset = load_dataset(key, "wikihow", cache_dir="./cache")
#         else:   
#             dataset = load_dataset(key, cache_dir="./cache")
#         if "train" in dataset:
#             train_dataset = dataset["train"]
#             train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in COLUMNS_TO_KEEP])

#             raw_train_datasets.append((train_dataset, value))

#         if "test" in dataset:
#             test_dataset = dataset["test"]
#             test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in COLUMNS_TO_KEEP])

#             raw_test_datasets.append(test_dataset)
#     train_subsets = []
#     for (dataset, frac) in raw_train_datasets:
#         train_subset = dataset.select(range(int(len(dataset)*frac)))
#         train_subsets.append(train_subset)
#     if shuffle:
#         train_dataset = concatenate_datasets(train_subsets).shuffle(seed=seed)
#     else:
#         train_dataset = concatenate_datasets(train_subsets)
#     if len(raw_test_datasets) > 0:
#         test_dataset = concatenate_datasets(raw_test_datasets).shuffle(seed=seed)
#     else:
#         test_dataset = None
    
#     new_dataset['train'] = train_dataset

#     if test_dataset is None:
#         new_dataset = new_dataset['train'].train_test_split(test_size=test_percentage)
#     else:
#         new_dataset['test'] = test_dataset
    
#     return new_dataset

def tokenize_function_and_group_texts(examples, tokenizer, block_size):
    EOS_TOKEN = tokenizer.eos_token
    texts = [text + EOS_TOKEN for text in examples["text"]]
    examples = tokenizer(texts)

    # examples['attention_mask'] = [[idx + 1] * len(value) for idx, value in enumerate(examples['attention_mask'])]
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result

def main():
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, ExtraConfig), ignore_extra_args=True)
    args, training_args, model_config, extra_args = parser.parse_args_and_config()
    # print(args)
    # print(training_args)
    # print(model_config)
    # print(extra_args)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(args.dataset_name):
        raw_datasets = load_from_disk(args.dataset_name)
    else:
        raw_datasets = load_dataset(args.dataset_name)
    print(raw_datasets)
    # print(extra_args.do_pretraining)
    if extra_args.do_pretraining is True:
        raw_datasets = raw_datasets.map(tokenize_function_and_group_texts, 
                                        fn_kwargs={"tokenizer": tokenizer, "block_size": training_args.max_seq_length }, 
                                        remove_columns=["text"],
                                        batched=True, 
                                        num_proc=extra_args.group_text_num_procs)

    train_dataset = raw_datasets[args.dataset_train_split]
    # eval_dataset = raw_datasets[args.dataset_test_split]
    try:
        eval_dataset = raw_datasets[args.dataset_test_split]
    except:
        dataset = train_dataset.train_test_split(test_size=0.001)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

    
    if extra_args.do_pretraining:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            tokenizer=tokenizer,
            packing=False,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=get_peft_config(model_config),
        )
    else:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
        )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

if __name__=='__main__':
    main()

     
