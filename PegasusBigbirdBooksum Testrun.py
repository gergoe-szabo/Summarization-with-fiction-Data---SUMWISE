#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline


# In[2]:


_model = AutoModelForSeq2SeqLM.from_pretrained(
                "pszemraj/bigbird-pegasus-large-K-booksum",
                low_cpu_mem_usage=True,
            )

_tokenizer = AutoTokenizer.from_pretrained(
                "pszemraj/bigbird-pegasus-large-K-booksum",
            )
                                           

summarizer = pipeline(
                    "summarization", 
                    model=_model, 
                    tokenizer=_tokenizer
                )


# In[4]:


with open('/mnt/data/users/szabo/Master_Thesis/HP1_BertSumExt.txt') as f:
    wall_of_text = f.read()

result = summarizer(
           wall_of_text,
           min_length=16, 
           max_length=256,
           no_repeat_ngram_size=3, 
           clean_up_tokenization_spaces=True,
    )

print(result[0]['summary_text'])


# In[ ]:


with open('/mnt/data/users/szabo/Master_Thesis/HP1_PegasusBigbirdLargeKBooksum.txt', 'w', encoding='utf-8') as t:
    t.write(full)

