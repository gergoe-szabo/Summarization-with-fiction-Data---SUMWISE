#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from datasets import load_dataset, Dataset
import pandas as pd
import json
from transformers import AutoTokenizer

with open('/Booksum/booksum_train.json', "r") as f:
    dataset = Dataset.from_dict(json.load(f))

len(dataset)

dataset = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=1024, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)


# In[ ]:


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")


# In[ ]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# In[ ]:


training_args = Seq2SeqTrainingArguments(
    output_dir="./results-PEG",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()



