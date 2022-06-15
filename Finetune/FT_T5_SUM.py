!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset, Dataset
import pandas as pd
import json
from transformers import AutoTokenizer


# In[ ]:


with open('/home/g-ball/Desktop/Projects/THESIS/Booksum/Dataset/booksum_train.json', "r", encoding="utf-8") as f:
    train_data = Dataset.from_dict(json.load(f))


# In[ ]:


with open('/home/g-ball/Desktop/Projects/THESIS/Booksum/Dataset/booksum_val.json', "r", encoding="utf-8") as f:
    val_data = Dataset.from_dict(json.load(f))


# In[ ]:


len_train = len(train_data)
print("entities in training dataset: ", len_train)
len_val = len(val_data)
print("entities in evaluation dataset: ", len_val)


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("t5-small")

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=1024, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_data = train_data.map(preprocess_function, batched=True)

tokenized_val_data = val_data.map(preprocess_function, batched=True)


# In[ ]:


from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# In[ ]:


training_args = Seq2SeqTrainingArguments(

    output_dir="./results-FN-T5-Small-Booksum",

    evaluation_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=12,

    per_device_eval_batch_size=12,

    weight_decay=0.01,

    save_total_limit=5,

    num_train_epochs=1,

    fp16=True,

)

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train_data,

    eval_dataset=tokenized_val_data,

    tokenizer=tokenizer,

    data_collator=data_collator,

)

trainer.train()
