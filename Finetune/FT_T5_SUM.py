#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from datasets import load_dataset

booksum = load_dataset("kmfoda/booksum")


# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")


# In[ ]:


prefix = "summarize: "


# In[ ]:


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["chapter"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary_text"], max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:


tokenized_booksum = booksum.map(preprocess_function, batched=True)



# In[ ]:


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


# In[ ]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# In[ ]:


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_booksum["train"],
    eval_dataset=tokenized_booksum["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


# In[ ]:




