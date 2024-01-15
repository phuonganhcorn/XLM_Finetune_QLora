from transformers import TrainingArguments, BitsAndBytesConfig, AutoTokenizer
from model.mrc_model2 import MRCQuestionAnswering
from transformers import Trainer
import transfomer
from utils import data_loader2
import numpy as np
from datasets import load_metric
import os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


#os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    model_id = "xlm-roberta-large"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='/home/phanh/Downloads/XLM-Finetune/XLM-Finetune/model-bin2/cache',
                                          #local_files_only=True
                                         )

    model = MRCQuestionAnswering.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0},cache_dir='/home/phanh/Downloads/XLM-Finetune/model-bin2/cache')
    model.to("cuda")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
        

    train_dataset, valid_dataset = data_loader2.get_dataloader(
        train_path='/home/phanh/Downloads/XLM-Finetune/data-bin/processed/train.dataset',
        valid_path='/home/phanh/Downloads/XLM-Finetune/data-bin/processed/valid.dataset'
    )
    
    #train_dataset = train_dataset.to("cuda")
    #valid_dataset = valid_dataset.to("cuda")
    
    '''
    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path='/content/XLM-Finetune/data-bin/processed/train.dataset',
        valid_path='/content/XLM-Finetune/data-bin/processed/valid.dataset'
    )
    '''
    
    training_args = TrainingArguments("/home/phanh/Downloads/XLM-Finetune/model-bin2/test",
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=2,
                                      learning_rate=2e-4,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      fp16 = True,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,
                                      gradient_accumulation_steps=1,
                                      logging_dir='./log',
                                      logging_steps=5,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      save_strategy="epoch",
                                      metric_for_best_model='f1',
                                      load_best_model_at_end=True,
                                      save_total_limit=2,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      evaluation_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_loader2.data_collator,
        compute_metrics=data_loader2.compute_metrics,
        #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    

    model.config.use_cache = False 
    trainer.train()
