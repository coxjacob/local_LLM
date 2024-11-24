# https://www.datacamp.com/tutorial/how-to-train-a-llm-with-pytorch

import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # deprecated prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(train_dataset)

pandas_format = train_dataset.to_pandas()
print(pandas_format.head())  #example used display, but that is ipython specific

import textwrap


# Note the below code prints the instruction, output, and text
# 
for index in range(2):
   print("---"*15)
   print("Instruction: {}".format(textwrap.fill(pandas_format.iloc[index]["instruction"],    
       width=50)))
   print("Output:  {}".format(textwrap.fill(pandas_format.iloc[index]["output"],  
       width=50)))
   print("Text: {}".format(textwrap.fill(pandas_format.iloc[index]["text"],  
       width=50)))
   
pretrained_model_name = "Salesforce/xgen-7b-8k-base"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

model_training_args = TrainingArguments(
       output_dir="xgen-7b-8k-base-fine-tuned",
       per_device_train_batch_size=4,
       optim="adamw_torch",
       logging_steps=80,
       learning_rate=2e-4,
       warmup_ratio=0.1,
       lr_scheduler_type="linear",
       num_train_epochs=1,
       save_strategy="epoch"
   )

import matplotlib.pyplot as plt


pandas_format['text_length'] = pandas_format['text'].apply(len)


plt.figure(figsize=(10,6))
plt.hist(pandas_format['text_length'], bins=50, alpha=0.5, color='g')
plt.title('Distribution of Length of Text')
plt.xlabel('Length of Text')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

mask = pandas_format['text_length'] > 1024
percentage = (mask.sum() / pandas_format['text_length'].count()) * 100


print(f"The percentage of text documents with a length greater than 1024 is: {percentage}%")

SFT_trainer = SFTTrainer(
       model=model,
       train_dataset=train_dataset,
       dataset_text_field="text",
       max_seq_length=1024,
       tokenizer=tokenizer,
       args=model_training_args,
       packing=True,
       peft_config=lora_peft_config,
   )

