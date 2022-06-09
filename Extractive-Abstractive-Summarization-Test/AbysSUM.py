#!/usr/bin/env python
# coding: utf-8

# In[3]:


from summarizer import Summarizer
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline


# In[4]:


#name = input("Enter the authors name: ")
name = "J.K. Rowling"

with open('HP1') as f:
    text = f.read()
text_no_num = re.sub(r'\d+','', text)   
body = text_no_num.replace("\n", " ").replace(name, "")
print(body)


# In[ ]:


text = body
text = text.split()
n = 1000
batches = [' '.join(text[i:i+n]) for i in range(0,len(text),n)]


# In[ ]:


word_list = batches[0].split()

number_of_words_summarized = len(word_list)

print(number_of_words_summarized)


# In[ ]:


hf_name = 'pszemraj/led-large-book-summary'

_model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_name,
                low_cpu_mem_usage=True,
            )

_tokenizer = AutoTokenizer.from_pretrained(
                hf_name
            )
                                           

summarizer = pipeline(
                    "summarization", 
                    model=_model, 
                    tokenizer=_tokenizer
                )


# In[ ]:


results = summarizer(
           batches,
           min_length=16, 
           max_length=256,
           no_repeat_ngram_size=3, 
           encoder_no_repeat_ngram_size =3,
           clean_up_tokenization_spaces=True,
           repetition_penalty=3.7,
           num_beams=4,
           early_stopping=True,
    )


# In[ ]:


abst_sum = results[0]['summary_text']


# In[ ]:


with open('AbysSUM-results.txt', 'w', encoding='utf-8') as t:
    t.write(abst_sum)

